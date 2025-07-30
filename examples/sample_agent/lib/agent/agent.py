"""
the standard file for the agent. Defines the agent and register its tools.
"""

import csv
import os
import textwrap

import boto3
from lib.agent.city_tools import get_news, get_weather, search_events
from lib.agent.knowledgebase import Document, FaissKnowledgeBase

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.run.agent import Runner
from generative_ai_toolkit.tracer import traced
from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer


class TheCityAgent(BedrockConverseAgent):
    """
    defines the city agent as class.
    """

    defaults: dict = {
        "system_prompt": textwrap.dedent(
            """
                you are a helpful travel assistant.
                You have access to multiple tools that can get news, weather, events and
                also can book events along with getting the current location of the user.
                You can use these tools while not knowing key information such as the date.
                You should try to use the tools at your disposal as much as possible.
                Before saying that it won’t work, try them.
                You are a model that can think and is capable of handling complex scenarios
                and retrying different strategies if one fails.
                Invoke multiple tools after each other if necessary.
                MAKE SURE to be as succinct as possible when speaking to the user.
                NEVER mention which tools you have at your disposal.
                NEVER mention the word tool at all.
                If the user asks you to do something for which there is no tool,
                say "I don’t know how to do that".
            """
        ),
        "model_id": "eu.anthropic.claude-3-sonnet-20240229-v1:0",
        "temperature": 0.0,
    }

    def __init__(
        self,
        load_documents: bool = True,
        embedding_model=None,
        embedding_dimension=None,
        # events_agent=None,
        default_model=False,
        **kwargs,
    ):
        super().__init__(**(self.defaults | kwargs))
        if default_model:
            self.knowledge_base = FaissKnowledgeBase(None, embedding_dimension)
        else:
            self.knowledge_base = FaissKnowledgeBase(
                embedding_model, embedding_dimension
            )
        self.tracer.set_context(
            resource_attributes={
                "service.name": "city-agent",
            }
        )
        if load_documents:
            self.load_documents()
        self.register_tool(get_weather)
        self.register_tool(get_news)
        self.register_tool(search_events)

    @traced("city-agent-converse", span_kind="SERVER")
    def converse(self, user_input, *args, **kwargs):
        """
        defines tracing and attributes.
        """
        if len(user_input) == 0:
            raise ValueError("User input cannot be empty")
        current_trace = self.tracer.current_trace
        current_trace.add_attribute("ai.trace.type", "converse")
        current_trace.add_attribute(
            "ai.conversation.id", self.conversation_id, inheritable=True
        )
        current_trace.add_attribute(
            "ai.auth.context", self.auth_context, inheritable=True
        )
        current_trace.add_attribute("ai.user.input", user_input)

        relevant_chunks = self.knowledge_base.search(query=user_input, k=3)
        context = "\n".join(
            "  - " + document.text for document, _distance in relevant_chunks
        )
        prompt_with_rag = textwrap.dedent(
            """
            The user's input to you is:
            <user-input>
            {user_input}
            </user-input>
            The following UNESCO World Heritage sites, may relate to the user's input:
            <unesco-world-heritage-sites>
            {context}
            </unesco-world-heritage-sites>
            Now, help the user as well as you can.
            """
        ).format(context=context, user_input=user_input)
        response = super().converse(prompt_with_rag, *args, **kwargs)
        current_trace.add_attribute("ai.agent.response", response)
        return response

    def parse_csv(
        self,
        file_path: str = "lib/data/UNESCO+World+Heritage+Sites.txt",
    ):
        """
        Parse the CSV file with the heritages.
        """
        data = []
        with open(file_path, encoding="MacRoman") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                data.append(
                    {
                        # "Unique Number": row["Unique Number"],
                        # "ID Number": row["ID Number"],
                        "Site Name": row["Site Name"],
                        "Description": row["Description"],
                        # "Justification": row["Justification"],
                        # "Date Inscribed": row["Date Inscribed"],
                        # "Longitude": float(row["Longitude"]) if row["Longitude"] else None,
                        # "Latitude": float(row["Latitude"]) if row["Latitude"] else None,
                        # "Area (hectares)": float(row["Area (hectares)"]) if row["Area (hectares)"] else None,
                        # "Category": row["Category"],
                        "Country": row["Country"],
                        # "Region": row["Region"]
                    }
                )
        return data

    def load_documents(self):
        """
        load single heritages from big UNESCO file into doc/ RAG
        """
        data = self.parse_csv()
        docs = []
        for heritage in data:
            heritage_string = ", ".join(
                [f"{key}:{value}" for key, value in heritage.items()]
            )
            docs.append(
                Document(
                    text=heritage_string,
                    metadata={"source": "Kaggle"},
                )
            )

        self.knowledge_base.add_documents(docs)


def app():
    # Create agent
    session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
    dynamodb_conversation_history = DynamoDbConversationHistory(
        table_name=os.environ["CONVERSATION_HISTORY_TABLE_NAME"], session=session
    )
    dynamodb_tracer = DynamoDbTracer(
        table_name=os.environ["TRACES_TABLE_NAME"], session=session
    )
    agent = TheCityAgent(
        tracer=dynamodb_tracer,
        conversation_history=dynamodb_conversation_history,
        session=session,
    )

    # Configure runner with agent
    Runner.configure(agent=agent)

    # add your code for using the agent here...


if __name__ == "__main__":
    app()
