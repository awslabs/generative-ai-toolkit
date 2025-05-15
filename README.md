# Generative AI Toolkit

The **Generative AI Toolkit** is a lightweight library that covers the life cycle of LLM-based applications, including agents. Its purpose is to support developers in building and operating high quality LLM-based applications, over their entire life cycle, starting with the very first deployment, in an automated workflow:

- **Build new agents** using any of the LLM's supported by the Amazon Bedrock Converse API. Streaming responses are supported, and you can simply add your Python functions as tools. Conversation history is stored in Amazon DynamoDB, other backends can be added easily. Agents can be deployed as AWS Lambda functions (exposed via Function URL), or as containers on Amazon ECS or EKS.
- **OpenTelemetry compatible tracing** to track in detail what your agents do. Out-of-the-box tracers for DynamoDB and AWS X-Ray are included, and it's easy to add custom tracers.
- **Evaluate your agents** with metrics, both while developing as well as in production––as your agents interact with users. Use the out-of-the-box metrics such as Cost, Latency, Cosine similarity, Conciseness, or add your own custom metrics. Metrics can be exported to Amazon CloudWatch Metrics easily, allowing you to tap into the full power of Amazon CloudWatch for observability.
- **Test your agents** on a deep level by writing assertions against the collected traces. Assert that your agents e.g. respond to users in the right way, and that they invoke the right tools, with the right inputs. A conversational mock of the Amazon Bedrock Converse API is included, so that you can make your tests deterministic and fast.

The toolkit builds upon principles and methodologies detailed in our research paper:

**[GENERATIVE AI TOOLKIT- A FRAMEWORK FOR INCREASING THE QUALITY OF LLM-BASED APPLICATIONS OVER THEIR WHOLE LIFE CYCLE](https://arxiv.org/abs/2412.14215)**.

## Highlights

The Generative AI Toolkit helps you build reliable agents, where you as developer have full visibility into what your agents do, both during development and in production; when your agents interact with real users. To this end, Generative AI Toolkit captures traces for all interactions between your users and your agents, and for the agent's internal actions, such as LLM invocations and tool invocations. With these traces you can use the Generative AI Toolkit to evaluate your agents continuously.

You can view the traces, as well as the collected metrics, in various developer friendly ways, e.g. with the web UI:

<img src="./assets/images/ui-conversation.png" alt="UI Conversation Display Screenshot" title="UI Conversation Display" width="1200"/>

Traces are also visible in AWS X-Ray (or any other OpenTelemetry compatible tool of your choosing):

<img src="./assets/images/x-ray-trace-map.png" alt="AWS X-Ray Trace Map Screenshot" title="AWS X-Ray Trace Map" width="1200"/>

Details of the traces, such as the LLM inputs and outputs, are visible in the trace timelines as metadata:

<img src="./assets/images/x-ray-trace-segments-timeline.png" alt="AWS X-Ray Trace Segments Timeline Screenshot" title="AWS X-Ray Trace Segments Timeline" width="1200"/>

Metrics can be emitted to Amazon CloudWatch easily, so you can create dashboards and alarms there, and tap into the full power of Amazon CloudWatch for observability:

<img src="./assets/images/sample-metric.png" alt="Sample Amazon Cloud Metric" width="1200" />

## Reference Architecture

The following is a reference architecture for a setup that uses the Generative AI Toolkit to implement an agent, collect traces, and run automated evaluation. The resulting metrics are fed back to the agent's developers via dashboards and alerts. Metrics are calculated and captured continuously, as real users interact with the agent, thereby giving the agent's developers insight into how the agent is performing at all times, allowing for continuous improvement:

<img src="./assets/images/architecture.drawio.png" alt="Architecture" width="1200" />

> Also see our **sample notebook [deploying_on_aws.ipynb](/examples/deploying_on_aws.ipynb)**.

## Key Terms

To fully utilize the Generative AI Toolkit, it’s essential to understand the following key terms:

- **Traces**: Traces are records of the internal operations of your LLM-based application, e.g. LLM invocations and tool invocations. Traces capture the entire request-response cycle, including input prompts, model outputs, tool calls, and metadata such as latency, token usage, and execution details. Traces form the foundation for evaluating an LLM-based application's behavior and performance.

- **Metrics**: Metrics are measurements derived from traces that evaluate various aspects of an LLM-based application's performance. Examples include latency, token usage, similarity with expected responses, sentiment, and cost. Metrics can be customized to measure specific behaviors or to enforce validation rules.

- **Cases**: Cases are repeatable test inputs that simulate conversations with the agent, e.g. for the purpose of agent evaluation. They consist of a sequence of user inputs and expected agent behaviors or outcomes. Cases are used to validate the agent's responses against defined expectations, ensuring consistent performance across scenarios.

- **Agents**: An agent is an implementation of an LLM-based application that processes user inputs and generates responses. The toolkit provides a simple and extensible agent implementation with built-in support for tracing and tool integration.

- **Tools**: Tools are external functions or APIs that agents can invoke to provide additional capabilities (e.g., fetching weather data or querying a database). Tools are registered with agents and seamlessly integrated into the conversation flow.

- **Conversation History**: This refers to the sequence of messages exchanged between the user and the agent. It can be stored in memory or persisted to external storage, such as DynamoDB, to maintain context across sessions.

- **CloudWatch Custom Metrics**: These are metrics logged to Amazon CloudWatch in Embedded Metric Format (EMF), enabling the creation of dashboards, alarms, and aggregations to monitor agent performance in production environments.

- **Web UI**: A local web-based interface that allows developers to inspect traces, debug conversations, and view evaluation results interactively. This is particularly useful for identifying and resolving issues in the agent's responses.

## Table of Contents

2.1 [Installation](#21-installation)  
2.2 [Agent Implementation](#22-agent-implementation)  
2.3 [Tracing](#23-tracing)  
2.4 [Evaluation Metrics](#24-evaluation-metrics)  
2.5 [Repeatable Cases](#25-repeatable-cases)  
2.6 [Cases with Dynamic Expectations](#26-cases-with-dynamic-expectations)  
2.7 [Generating Traces: Running Cases in Bulk](#27-generating-traces-running-cases-in-bulk)  
2.8 [CloudWatch Custom Metrics](#28-cloudwatch-custom-metrics)  
2.9 [Deploying and Invoking the BedrockConverseAgent](#29-deploying-and-invoking-the-bedrockconverseagent)  
2.10 [Web UI for Conversation Debugging](#210-web-ui-for-conversation-debugging)  
2.11 [Mocking and Testing](#211-mocking-and-testing)  
2.12 [Model Context Protocol (MCP) Client](#212-model-context-protocol-mcp-client)

### 2.1 Installation

Install `generative_ai_toolkit` with support for all features, amongst which interactive evaluation of metrics:

```bash
pip install "generative-ai-toolkit[all]"  # Note the [all] modifier
```

If you don't use the `[all]` installation modifier, only the minimal set of dependencies will be included that you'll need for creating an agent.

Other available modifiers are:

- `[run-agent]`: includes dependencies such as `gunicorn` that allow you to use `generative_ai_toolkit.run.agent.Runner` to expose your agent over HTTP.
- `[evaluate]`: includes dependencies that allow you to run evaluations against traces.

### 2.2 Agent implementation

The heart of the Generative AI Toolkit are the traces it collects, that are the basis for evaluations (explained below). The toolkit includes a simple agent implementation that is backed by the [Amazon Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) and that is instrumented to collect traces in the right format.

A benefit of using this agent implementation, is that you can run the agent locally––it doesn't require any AWS deployment at all and only needs Amazon Bedrock model access. You can quickly iterate and try different agent settings, such as the backing LLM model id, system prompt, temperature, tools, etc. You can create repeatable test cases and run extensive and rigorous evaluations locally.

> We'll first explain how our agent implementation works. Feel free to directly skip to the explanation of [Tracing](#23-tracing) or [Metrics](#24-metrics) instead.

The Generative AI Toolkit Agent implementation is simple and lightweight, and makes for a no-nonsense developer experience. You can easily instantiate and converse with agents while working in the Python interpreter (REPL) or in a notebook:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
```

Obviously right now this agent doesn't have any tools yet (we'll add some shortly), but you can already chat with it.

#### Chat with agent

Use `converse()` to chat with the agent. You pass the user's input to this function, and it will return the agent's response as string:

```python
response = agent.converse("What's the capital of France?")
print(response) # "The capital of France is Paris."
```

#### Response streaming

You can also use `converse_stream()` to chat with the agent. You pass the user's input to this function, and it will return an iterator that will progressively return the response fragments. You should concatenate these fragments to collect the full response.

The benefit over using `converse()` is that you can show the user the agent's response tokens as they're being generated, instead of only showing the full response at the very end:

```python
for fragment in agent.converse_stream("What's the capital of France?"):
    print(fragment)
```

That example might now print several lines to the console, for each set of tokens received, e.g.:

```
The
 capital
 of France is
 Paris.
```

#### Conversation history

The agent maintains the conversation history, so e.g. after the question just asked, this would now work:

```python
response = agent.converse("What are some touristic highlights there?") # This goes back to what was said earlier in the conversation
print(response) # "Here are some of the major tourist highlights and attractions in Paris, France:\n\n- Eiffel Tower - One of the most famous monuments ..."
```

By default conversation history is stored in memory only. If you want to use conversation history across different process instantiations, you need conversation history that is persisted to durable storage.

#### Persisting conversation history

You can use the `DynamoDbConversationHistory` class to persist conversations to DynamoDB. Conversation history is maintained per conversation ID. The agent will create a new conversation ID automatically:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    conversation_history=DynamoDbConversationHistory(table_name="conversations") # This table needs to exist, with string keys "pk" and "sk"
)

print(agent.conversation_id) # e.g.: "01J5D9ZNK5XKZX472HC81ZYR5P"

agent.converse("What's the capital of France?") # This message, and the agent's response, will now be stored in DynamoDB under conversation ID "01J5D9ZNK5XKZX472HC81ZYR5P"
```

Then later, in another process, if you want to continue this conversation, set the conversation ID first:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    conversation_history=DynamoDbConversationHistory(table_name="conversations")
)

agent.set_conversation_id("01J5D9ZNK5XKZX472HC81ZYR5P")

response = agent.converse("What are some touristic highlights there?")
print(response) # "Here are some of the major tourist highlights and attractions in Paris, France:\n\n- Eiffel Tower - One of the most famous monuments ..."
```

#### Viewing the conversation history

You can manually view the conversation history like so:

```python
print(agent.messages)
# [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}, {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}, {'role': 'user', 'content': [{'text': 'What are some touristic ...
```

Conversation history is included automatically in the prompt to the LLM. That is, you only have to provide new user input when you call `converse()` (or `converse_stream()`), but under the hood the agent will include all past messages as well.

This is generally how conversations with LLMs work––the LLM has no memory of the current conversation, you need to provide all past messages, including those from the LLM (the "assistant"), as part of your prompt to the LLM.

#### Starting a fresh conversation

Calling `agent.reset()` starts a new conversation, with empty conversation history:

```python
print(agent.conversation_id)  # e.g.: "01J5D9ZNK5XKZX472HC81ZYR5P"
agent.converse("Hi!")
print(len(agent.messages)) # 2 (user input + agent response)
agent.reset()
print(len(agent.messages)) # 0
print(agent.conversation_id)  # e.g.: "01J5DQRD864TR3BF314CZK8X5B" (changed)
```

#### Bedrock Converse Arguments

Upon instantiating the `BedrockConverseAgent` you can pass any arguments that the Bedrock Converse API accepts, and these will be used for all invocations of the Converse API by the agent. You could for example specify usage of [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html):

```python
from generative_ai_toolkit.agent import BedrockConverseAgent

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    system_prompt="your system prompt",
    max_tokens=500,
    temperature=0.0,
    top_p=0.8,
    stop_sequences=["stop sequence"],
    guardrail_identifier="guardrail-id",
    guardrail_version="guardrail-version",
    guardrail_trace="enabled_full",
    guardrail_stream_processing_mode="async",
    additional_model_request_fields={"foo": "bar"},
    prompt_variables={"foo": {"text": "bar"}},
    additional_model_response_field_paths=["/path"],
    request_metadata={"foo": "bar"},
    performance_config={"latency": "optimized"},
)
```

##### Reasoning

If you want to use reasoning with a model that supports it (e.g. `anthropic.claude-3-7-sonnet-20250219-v1:0`), specify `additional_model_request_fields`:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    additional_model_request_fields={
        "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
    },
)
```

Then, when calling `converse` or `converse_stream`, reasoning texts will be included within `<thinking>` tags in the output:

```python
response = agent.converse("How should I make Spaghetti Carbonara?")
print(response)
```

Would print e.g.:

```
<thinking>
The user is asking for a recipe for Spaghetti Carbonara. I have a tool available called `get_recipe` that can provide recipes.

The required parameter for this function is:
- dish: The name of the dish to get a recipe for

In this case, the dish is "Spaghetti Carbonara". This is clearly stated in the user's request, so I can call the function with this parameter.
</thinking>

I can help you with a recipe for Spaghetti Carbonara! Let me get that for you.
Here is the recipe ...
```

If you do not want to include the reasoning texts in the output, you can turn that off like so:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    additional_model_request_fields={
        "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
    },
    include_reasoning_text_within_thinking_tags=False, # set this to False
)
```

#### Tools

If you want to give the agent access to tools, you can define them as Python functions, and register them with the agent. Your Python function must have type annotations for input and output, and a docstring like so:

```python
def weather_report(city_name: str) -> str:
    """
    Gets the current weather report for a given city

    Parameters
    ------
    city_name: string
      The name of the city
    """
    return "Sunny" # return a string, number, dict or list --> something that can be turned into JSON

agent.register_tool(weather_report)

response = agent.converse("What's the weather like right now in Amsterdam?")
print(response) # Okay, let me get the current weather report for Amsterdam using the available tool: The weather report for Amsterdam shows that it is currently sunny there.
```

As you can see, tools that you've registered will be invoked automatically by the agent. The output from `converse` is always just a string with the agent's response to the user.

#### Other tools

If you don't want to register a Python function as tool, but have a tool with tool spec ready, you can also use it directly, as long as your tool satisfies the `Tool` protocol, i.e. has this shape:

```python
from typing import Any
from mypy_boto3_bedrock_runtime.type_defs import ToolSpecificationTypeDef

class MyTool:

    @property
    def tool_spec(self) -> ToolSpecificationTypeDef:
        return {"name":"my-tool","description":"This tool helps with ...", "inputSchema": {...}}

    def invoke(self, *args, **kwargs) -> Any:
        return "Tool response"

agent.register_tool(MyTool())
```

It's also possible to provide the tool spec explicitly alongside your plain Python function:

```python
agent.register_tool(
    lambda preferred_weather: f"Not {preferred_weather}",
    tool_spec={
        "name": "get_weather",
        "description": "Gets the current weather",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "preferred_weather": {
                        "type": "string",
                        "description": "The preferred weather",
                    },
                },
                "required": [
                    "preferred_weather",
                ],
            }
        },
    },
)
```

#### Tools override

It's possible to set and override the tool selection when calling converse:

```python
def bad_weather_report(city_name: str) -> str:
    """
    Gets the current weather report for a given city

    Parameters
    ------
    city_name: string
      The name of the city
    """
    return "Rainy"

response = agent.converse("What's the weather like right now in Amsterdam?", tools=[bad_weather_report])
print(response) # Okay, let me check the current weather report for Amsterdam using the available tool:\nAccording to the tool, the current weather report for Amsterdam is rainy.
```

Note that this does not force the agent to use the provided tools, it merely makes them available for the agent to use.

### 2.3 Tracing

You can make `BedrockConverseAgent` log traces of the LLM and tool calls it performs, by providing a tracer class.

In the following example, the `InMemoryTracer` is used, which is meant for use during development:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.tracer import InMemoryTracer # Import tracer

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    conversation_history=DynamoDbConversationHistory(table_name="conversations"),
    tracer=InMemoryTracer, # Add tracer
)
```

Now, when you `converse()` with the agent, and the agent calls e.g. LLM and tools, it will log traces. You can inspect these traces like so:

```python
response = agent.converse("What's the capital of France?")
print(agent.traces[0])
```

Will output e.g.:

```python
Trace(span_name='converse', span_kind='SERVER', trace_id='33185be48ee341d16bf681a552535a4a', span_id='935272e82e76823c', parent_span_id=None, started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 961, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 715109, tzinfo=datetime.timezone.utc), attributes={'ai.trace.type': 'converse', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None, 'ai.user.input': "What's the capital of France?", 'ai.agent.response': 'The capital of France is Paris.'}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)
```

That is the root trace of the conversation, that captures user input and agent response. Other traces capture details such as LLM invocations, Tool invocations, usage of conversation history, etc.

#### Available tracers

The Generative AI Toolkit includes several tracers out-of-the-box, e.g. the `DynamoDBTracer` that saves traces to DynamoDB, and the `OtlpTracer` that sends traces to an OpenTelemetry collector (e.g. to forward them to AWS X-Ray).

For a full run-down of all out-of-the-box tracers and how to use them, view [examples/tracing101.ipynb](examples/tracing101.ipynb).

#### Open Telemetry

Traces use the [OpenTelemetry "Span" model](https://opentelemetry.io/docs/specs/otel/trace/api/#span). That model works at high level by assigning a unique Trace ID to each incoming request (e.g. over HTTP). All actions that are taken while executing that request, are recorded as "span" and will have a unique Span ID. Span name, start timestamp, end timestamp, are recorded at span level.

So, for example, when a user sends a message to an agent, that will start a trace. Then, for every action the agent takes to handle the user's request, a span is recorded. All these spans share the same trace ID, but have a unique span ID. For example, if the agent invokes an LLM or tool, that is recorded as a span. When the agent returns the response to the user, the trace ends, and multiple spans will have been recorded. Often, user and agent will have a conversation that includes multiple turns: the user gives the agent an instruction, the agent asks follow up questions or confirmation, the user gives additional directions, and so forth until the user's intent is fully achieved. Each back-and-forth between user and agent, i.e. each turn in the conversation, is a trace and will have a unique trace ID and (likely) include multiple spans.

In the OpenTelemetry Span model, information such as "the model ID used for the LLM invocation" must be added to a span as attributes. The Generative AI Toolkit uses the following span attributes:

| Attribute Name                                         | Description                                                                                                                                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ai.trace.type`                                        | Used to identify the type of trace operation being performed. Values: "conversation-history-list", "conversation-history-add", "converse", "converse-stream", "tool-invocation", "llm-invocation" |
| `ai.conversation.history.implementation`               | The string representation of the conversation history implementation being used (e.g. the name of the Python class)                                                                               |
| `peer.service`                                         | Indicates the service being interacted with. Values: "memory:short-term", "tool:{tool_name}", "llm:{model_id}"                                                                                    |
| `ai.conversation.history.messages`                     | Contains the messages from the conversation history                                                                                                                                               |
| `ai.conversation.history.message`                      | Contains a single message being added to the conversation history                                                                                                                                 |
| `ai.conversation.id`                                   | The unique identifier for the conversation (inheritable attribute)                                                                                                                                |
| `ai.auth.context`                                      | The authentication context for the conversation (inheritable attribute)                                                                                                                           |
| `ai.tool.name`                                         | Name of the tool being invoked                                                                                                                                                                    |
| `ai.tool.use.id`                                       | Unique identifier for the tool usage                                                                                                                                                              |
| `ai.tool.input`                                        | The input parameters provided to the tool                                                                                                                                                         |
| `ai.tool.output`                                       | The response/output from the tool invocation                                                                                                                                                      |
| `ai.tool.error`                                        | Error information if tool invocation fails                                                                                                                                                        |
| `ai.user.input`                                        | The input provided by the user in the conversation                                                                                                                                                |
| `ai.llm.request.inference.config`                      | Configuration settings for the LLM inference                                                                                                                                                      |
| `ai.llm.request.messages`                              | Messages being sent to the LLM                                                                                                                                                                    |
| `ai.llm.request.model.id`                              | Identifier of the LLM model being used                                                                                                                                                            |
| `ai.llm.request.system`                                | System prompt or configuration being sent to the LLM                                                                                                                                              |
| `ai.llm.request.tool.config`                           | Tool configuration being sent to the LLM                                                                                                                                                          |
| `ai.llm.request.guardrail.config`                      | Configuration for a guardrail applied during the request. It restricts or modifies the content in messages based on configured criteria.                                                          |
| `ai.llm.request.additional.model.request.fields`       | Additional inference parameters specific to the chosen model that extend the standard inference configuration options.                                                                            |
| `ai.llm.request.additional.model.response.field.paths` | Specifies additional fields from the model's response to include explicitly, identified by JSON Pointer paths.                                                                                    |
| `ai.llm.request.prompt.variables`                      | Variables defined in a prompt resource, mapped to values provided at runtime, used to dynamically customize prompts.                                                                              |
| `ai.llm.request.request.metadata`                      | Custom key-value pairs included for metadata purposes, primarily for filtering and analyzing invocation logs.                                                                                     |
| `ai.llm.request.performance.config`                    | Configuration that specifies performance-related settings, such as latency and resource allocation, tailored for specific model invocations.                                                      |
| `ai.llm.response.output`                               | Output received from the LLM                                                                                                                                                                      |
| `ai.llm.response.stop.reason`                          | Reason why the LLM stopped generating                                                                                                                                                             |
| `ai.llm.response.usage`                                | Usage metrics from the LLM response                                                                                                                                                               |
| `ai.llm.response.metrics`                              | Additional metrics from the LLM response                                                                                                                                                          |
| `ai.llm.response.error`                                | Error information if the LLM request fails                                                                                                                                                        |
| `ai.llm.response.trace`                                | Trace information                                                                                                                                                                                 |
| `ai.llm.response.performance.config`                   | The performance config                                                                                                                                                                            |
| `ai.agent.response`                                    | The final concatenated response from the agent                                                                                                                                                    |
| `service.name`                                         | Name of the service, set to the class name of the agent                                                                                                                                           |

#### Viewing traces

```python
for trace in agent.traces:
    print(trace)
    print()
```

Would e.g. print:

```python
Trace(span_name='converse', span_kind='SERVER', trace_id='33185be48ee341d16bf681a552535a4a', span_id='935272e82e76823c', parent_span_id=None, started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 961, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 715109, tzinfo=datetime.timezone.utc), attributes={'ai.trace.type': 'converse', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None, 'ai.user.input': "What's the capital of France?", 'ai.agent.response': 'The capital of France is Paris.'}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)

Trace(span_name='conversation-history-add', span_kind='CLIENT', trace_id='33185be48ee341d16bf681a552535a4a', span_id='ec7c8e79daac9be0', parent_span_id='935272e82e76823c', started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 1059, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 158808, tzinfo=datetime.timezone.utc), attributes={'ai.trace.type': 'conversation-history-add', 'ai.conversation.history.message': {'role': 'user', 'content': [{'text': "What's the capital of France?"}]}, 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)

Trace(span_name='conversation-history-list', span_kind='CLIENT', trace_id='33185be48ee341d16bf681a552535a4a', span_id='f23f49c975823d9d', parent_span_id='935272e82e76823c', started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 158828, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 186879, tzinfo=datetime.timezone.utc), attributes={'ai.trace.type': 'conversation-history-list', 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.history.messages': [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}], 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)

Trace(span_name='llm-invocation', span_kind='CLIENT', trace_id='33185be48ee341d16bf681a552535a4a', span_id='92ff8f46baa35ec1', parent_span_id='935272e82e76823c', started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 186905, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 686732, tzinfo=datetime.timezone.utc), attributes={'peer.service': 'llm:claude-3-sonnet', 'ai.trace.type': 'llm-invocation', 'ai.llm.request.inference.config': {}, 'ai.llm.request.messages': [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}], 'ai.llm.request.model.id': 'anthropic.claude-3-sonnet-20240229-v1:0', 'ai.llm.request.system': None, 'ai.llm.request.tool.config': None, 'ai.llm.response.output': {'message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}}, 'ai.llm.response.stop.reason': 'end_turn', 'ai.llm.response.usage': {'inputTokens': 14, 'outputTokens': 10, 'totalTokens': 24}, 'ai.llm.response.metrics': {'latencyMs': 350}, 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)

Trace(span_name='conversation-history-add', span_kind='CLIENT', trace_id='33185be48ee341d16bf681a552535a4a', span_id='f9e6c4ff0254811c', parent_span_id='935272e82e76823c', started_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 686771, tzinfo=datetime.timezone.utc), ended_at=datetime.datetime(2025, 4, 15, 19, 33, 38, 715055, tzinfo=datetime.timezone.utc), attributes={'ai.trace.type': 'conversation-history-add', 'ai.conversation.history.message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}, 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, span_status='UNSET', resource_attributes={'service.name': 'BedrockConverseAgent'}, scope=generative-ai-toolkit@current)

```

You can also display traces in a human friendly format:

```python
for trace in agent.traces:
    print(trace.as_human_readable())
```

Which would print e.g.:

```
[33185be48ee341d16bf681a552535a4a/root/935272e82e76823c] BedrockConverseAgent SERVER 2025-04-15T19:33:38.000Z - converse (ai.trace.type='converse' ai.conversation.id='01JRXF2JHXACD860A6P7N0MXER' ai.auth.context='null')
       Input: What's the capital of France?
    Response: The capital of France is Paris.

[33185be48ee341d16bf681a552535a4a/935272e82e76823c/ec7c8e79daac9be0] BedrockConverseAgent CLIENT 2025-04-15T19:33:38.001Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRXF2JHXACD860A6P7N0MXER' ai.auth.context='null')
     Message: {'role': 'user', 'content': [{'text': "What's the capital of France?"}]}

[33185be48ee341d16bf681a552535a4a/935272e82e76823c/f23f49c975823d9d] BedrockConverseAgent CLIENT 2025-04-15T19:33:38.158Z - conversation-history-list (ai.trace.type='conversation-history-list' peer.service='memory:short-term' ai.conversation.id='01JRXF2JHXACD860A6P7N0MXER' ai.auth.context='null')
    Messages: [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}]

[33185be48ee341d16bf681a552535a4a/935272e82e76823c/92ff8f46baa35ec1] BedrockConverseAgent CLIENT 2025-04-15T19:33:38.186Z - llm-invocation (ai.trace.type='llm-invocation' peer.service='llm:claude-3-sonnet' ai.conversation.id='01JRXF2JHXACD860A6P7N0MXER' ai.auth.context='null')
Last message: [{'text': "What's the capital of France?"}]
    Response: {'message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}}

[33185be48ee341d16bf681a552535a4a/935272e82e76823c/f9e6c4ff0254811c] BedrockConverseAgent CLIENT 2025-04-15T19:33:38.686Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRXF2JHXACD860A6P7N0MXER' ai.auth.context='null')
     Message: {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}

```

Or, as dictionaries:

```python
for trace in agent.traces:
    print(trace.as_dict())
    print()
```

Which would print e.g.:

```python
{'span_name': 'converse', 'span_kind': 'SERVER', 'trace_id': '33185be48ee341d16bf681a552535a4a', 'span_id': '935272e82e76823c', 'parent_span_id': None, 'started_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 961, tzinfo=datetime.timezone.utc), 'ended_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 715109, tzinfo=datetime.timezone.utc), 'attributes': {'ai.trace.type': 'converse', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None, 'ai.user.input': "What's the capital of France?", 'ai.agent.response': 'The capital of France is Paris.'}, 'span_status': 'UNSET', 'resource_attributes': {'service.name': 'BedrockConverseAgent'}, 'scope': {'name': 'generative-ai-toolkit', 'version': 'current'}}

{'span_name': 'conversation-history-add', 'span_kind': 'CLIENT', 'trace_id': '33185be48ee341d16bf681a552535a4a', 'span_id': 'ec7c8e79daac9be0', 'parent_span_id': '935272e82e76823c', 'started_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 1059, tzinfo=datetime.timezone.utc), 'ended_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 158808, tzinfo=datetime.timezone.utc), 'attributes': {'ai.trace.type': 'conversation-history-add', 'ai.conversation.history.message': {'role': 'user', 'content': [{'text': "What's the capital of France?"}]}, 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, 'span_status': 'UNSET', 'resource_attributes': {'service.name': 'BedrockConverseAgent'}, 'scope': {'name': 'generative-ai-toolkit', 'version': 'current'}}

{'span_name': 'conversation-history-list', 'span_kind': 'CLIENT', 'trace_id': '33185be48ee341d16bf681a552535a4a', 'span_id': 'f23f49c975823d9d', 'parent_span_id': '935272e82e76823c', 'started_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 158828, tzinfo=datetime.timezone.utc), 'ended_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 186879, tzinfo=datetime.timezone.utc), 'attributes': {'ai.trace.type': 'conversation-history-list', 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.history.messages': [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}], 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, 'span_status': 'UNSET', 'resource_attributes': {'service.name': 'BedrockConverseAgent'}, 'scope': {'name': 'generative-ai-toolkit', 'version': 'current'}}

{'span_name': 'llm-invocation', 'span_kind': 'CLIENT', 'trace_id': '33185be48ee341d16bf681a552535a4a', 'span_id': '92ff8f46baa35ec1', 'parent_span_id': '935272e82e76823c', 'started_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 186905, tzinfo=datetime.timezone.utc), 'ended_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 686732, tzinfo=datetime.timezone.utc), 'attributes': {'peer.service': 'llm:claude-3-sonnet', 'ai.trace.type': 'llm-invocation', 'ai.llm.request.inference.config': {}, 'ai.llm.request.messages': [{'role': 'user', 'content': [{'text': "What's the capital of France?"}]}], 'ai.llm.request.model.id': 'anthropic.claude-3-sonnet-20240229-v1:0', 'ai.llm.request.system': None, 'ai.llm.request.tool.config': None, 'ai.llm.response.output': {'message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}}, 'ai.llm.response.stop.reason': 'end_turn', 'ai.llm.response.usage': {'inputTokens': 14, 'outputTokens': 10, 'totalTokens': 24}, 'ai.llm.response.metrics': {'latencyMs': 350}, 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, 'span_status': 'UNSET', 'resource_attributes': {'service.name': 'BedrockConverseAgent'}, 'scope': {'name': 'generative-ai-toolkit', 'version': 'current'}}

{'span_name': 'conversation-history-add', 'span_kind': 'CLIENT', 'trace_id': '33185be48ee341d16bf681a552535a4a', 'span_id': 'f9e6c4ff0254811c', 'parent_span_id': '935272e82e76823c', 'started_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 686771, tzinfo=datetime.timezone.utc), 'ended_at': datetime.datetime(2025, 4, 15, 19, 33, 38, 715055, tzinfo=datetime.timezone.utc), 'attributes': {'ai.trace.type': 'conversation-history-add', 'ai.conversation.history.message': {'role': 'assistant', 'content': [{'text': 'The capital of France is Paris.'}]}, 'ai.conversation.history.implementation': 'DynamoDbConversationHistory(table_name=conversations, identifier=None)', 'peer.service': 'memory:short-term', 'ai.conversation.id': '01JRXF2JHXACD860A6P7N0MXER', 'ai.auth.context': None}, 'span_status': 'UNSET', 'resource_attributes': {'service.name': 'BedrockConverseAgent'}, 'scope': {'name': 'generative-ai-toolkit', 'version': 'current'}}

```

#### Web UI

You can view the traces for a conversation using the Generative AI Toolkit Web UI:

```python
from generative_ai_toolkit.ui import traces_ui
demo = traces_ui(agent.traces)
demo.launch()
```

That opens the Web UI at http://127.0.0.1:7860. E.g. a conversation, that includes an invocation of a weather tool, would look like this:

<img src="./assets/images/ui-traces.png" alt="UI Traces Display Screenshot" title="UI Traces Display" width="1000"/>

Note that by default only traces for LLM invocations and Tool invocations are shown, as well as user input and agent output. You can choose to view all traces, which would also show e.g. usage of conversational memory, and any other traces the agent developer may have decided to add.

Stop the Web UI as follows:

```python
demo.close()
```

#### DynamoDB example

As example, here's some traces that were stored with the `DynamoDBTracer`:

<img src="./assets/images/dynamodb-traces.png" alt="DynamoDB Traces Display Screenshot" title="DynamoDB Traces Display" width="1200"/>

In production deployments, you'll likely want to use the `DynamoDBTracer`, so you can listen to the DynamoDB stream as traces are recorded, and run metric evaluations against them (see next section). This way, you can monitor the performance of your agent in production.

#### AWS X-Ray example

Here's a more elaborate example of a set traces when viewed in AWS X-Ray (you would have used the `OtlpTracer` to send them there):

<img src="./assets/images/x-ray-trace-map.png" alt="AWS X-Ray Trace Map Screenshot" title="AWS X-Ray Trace Map" width="1200"/>

The AWS X-Ray view is great because it gives developers an easy-to-digest graphical representation of traces. It's easy to see what the agent did, in which order, how long these actions took, and what the trace attributes are that capture e.g. inputs and outputs for LLM invocations and tool invocations (see the "Metadata" pane on the right) :

<img src="./assets/images/x-ray-trace-segments-timeline.png" alt="AWS X-Ray Trace Segments Timeline Screenshot" title="AWS X-Ray Trace Segments Timeline" width="1200"/>

### 2.4 Evaluation Metrics

Metrics allow you to evaluate your LLM-based application (/agent). The Generative AI Toolkit comes with some metrics out of the box, and makes it easy to develop your own metric as well. Metrics work off of traces, and can measure anything that is represented within the traces.

Here is how you can run metrics against traces.

> Note, this is a contrived example for now; in reality you likely won't run metrics against a single conversation you had with the agent, but against a suite of test cases. Hold tight, that will be explained further below.

```python
from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit
from generative_ai_toolkit.metrics.modules.conciseness import AgentResponseConcisenessMetric
from generative_ai_toolkit.metrics.modules.latency import LatencyMetric

results = GenerativeAIToolkit.eval(
    metrics=[AgentResponseConcisenessMetric(), LatencyMetric()],
    traces=[agent.traces] # pass the traces that were automatically collected by the agent in your conversation with it
)

results.summary() # this prints a table with averages to stdout
```

Would e.g. print:

```
+-----------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
| Avg Conciseness | Avg Latency LLM | Avg Latency TOOL | Avg Latency get_weather | Avg Trace count per run | Avg LLM calls per run | Avg Tool calls per run | Total Nr Passed | Total Nr Failed |
+-----------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
|       8.0       |     1187.0      |       0.0        |           0.0           |           3.0           |          2.0          |          1.0           |        0        |        0        |
+-----------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
```

You can also access each individual measurement object:

```python
for conversation_measurements in results:
    for measurement in conversation_measurements.measurements:
        print(measurement) # measurement concerning all traces in the conversation
    for trace_measurements in conversation_measurements.traces:
        for measurement in trace_measurements.measurements:
            print(measurement) # measurement concerning an individual trace
```

Or, access the measurements as a (flattened) DataFrame:

```python
df = results.details()
df.head()
```

Note that measurements can easily be exported to Amazon CloudWatch as Custom Metrics, which allow you to use Amazon CloudWatch for creating dashboards, aggregations, alarms, etc. See further below.

#### Included metrics

The following metric are included in the Generative AI Toolkit out-of-the-box.

> Note that some of these metrics can only meaningfully be run during development, because they rely on developer expressed expectations (similar to expectations in a unit test). Developers can express these expectations in cases, explained further below.

| Class name                                                   | Description                                                                                                                                                                                                                                                                                                                            | Usage                   |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `metrics.modules.latency.TokensMetric`                       | Measures number of tokens in LLM invocations (input, output, total)                                                                                                                                                                                                                                                                    | Development, production |
| `metrics.modules.similarity.AgentResponseSimilarityMetric`   | Measures the cosine similarity between an agent's actual response, and the expected responses that were expressed in the case by the developer. This metric requires cases to have the property `expected_agent_responses_per_turn` specified, which can be provided either during instantiation of the case or with `case.add_turn()` | Development only        |
| `metrics.modules.bleu.BleuMetric`                            | Similar to the `AgentResponseSimilarityMetric`, but calculates the Bleu score to determine similarity, rather than using cosine similarity                                                                                                                                                                                             | Development only        |
| `metrics.modules.sentiment.SentimentMetric`                  | Measures the sentiment of the conversation, using Amazon Comprehend.                                                                                                                                                                                                                                                                   | Development, production |
| `metrics.modules.latency.LatencyMetric`                      | Measures the latency of LLM and Tool invocations                                                                                                                                                                                                                                                                                       | Development, production |
| `metrics.modules.cost.CostMetric`                            | Measures the cost of LLM invocations                                                                                                                                                                                                                                                                                                   | Development, production |
| `metrics.modules.conversation.ConversationExpectationMetric` | Measures how well the conversation aligns with overall expectations that were expressed by the developer in the case. This metric requires cases to have the property `overall_expectations` which can be provided during instantiation of the case.                                                                                   | Development only        |
| `metrics.modules.conciseness.AgentResponseConcisenessMetric` | Measures how concise the agent's response are, i.e. to aid in building agents that don't ramble. This metric is implemented as an LLM-as-judge: an LLM is used to grade the conciseness of the agent's response on a scale from 1 to 10.                                                                                               | Development, production |

#### Custom metrics

Let's now see how you create a custom metric. Here is a custom metric that would measure how many tools the agent actually used in the conversation with the user:

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class NumberOfToolsUsedMetric(BaseMetric):
    def evaluate_conversation(self, conversation_traces, **kwargs):
        return Measurement(
            name="NumberOfToolsUsed",
            value=len([trace for trace in conversation_traces if trace.attributes.get("ai.trace.type") == "tool-invocation"]),
            unit=Unit.Count,
        )
```

The above metric works at conversation level and therefore implements `evaluate_conversation` which gets all the traces from the conversation in one go.

Even more simple custom metrics would work at individual trace level, without needing to know about the other traces in the conversation. In that case, implement `evaluate_trace`:

> Note the `TokensMetric` actually comes out-of-the-box, but we'll reimplement it here for sake of the example

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class TokenCount(BaseMetric):
    if trace.attributes.get("ai.trace.type") != "llm-invocation":
        return

    input_tokens = trace.attributes["ai.llm.response.usage"]["inputTokens"]
    output_tokens = trace.attributes["ai.llm.response.usage"]["outputTokens"]

    return [
        Measurement(
            name="TotalTokens",
            value=input_tokens + output_tokens,
            unit=Unit.Count,
        ),
        Measurement(
            name="InputTokens",
            value=input_tokens,
            unit=Unit.Count,
        ),
        Measurement(
            name="OutputTokens",
            value=output_tokens,
            unit=Unit.Count,
        ),
    ]
```

The above custom metric returns 3 measurements, but only for LLM traces.

Evaluating your own custom metrics works the same as for the out-of-the-box metrics (and they can be matched freely):

```python
results = GenerativeAIToolkit.eval(
    metrics=[NumberOfToolsUsedMetric(), TokenCount()],
    traces=[agent.traces]
)
results.summary()
```

Would e.g. print:

```
+------------------------------+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
| Avg NumberOfToolsUsed        | Avg NrOfOInputTokens | Avg NrOfOutputTokens | Avg Trace count per run | Avg LLM calls per run | Avg Tool calls per run | Total Nr Passed | Total Nr Failed |
+------------------------------+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
|             1.0              |        371.0         |         42.5         |           3.0           |          2.0          |          1.0           |        0        |        0        |
+------------------------------+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
```

#### Template for Custom Metrics

Use [TEMPLATE_metric.py](src/generative_ai_toolkit/metrics/modules/TEMPLATE_metric.py) as a starting point for creating your own custom metrics. This file includes more information on the data model, as well as more examples.

#### Passing or Failing a Custom Metric

Besides measuring an agent's performance in a scalar way, custom metrics can (optionally) return a Pass or Fail indicator. This will be reflected in the measurements summary and such traces would be marked as failed in the Web UI for conversation debugging (see further).

Let's tweak our `TokenCount` metric to make it fail if the LLM returns more than 100 tokens:

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class TokenCount(BaseMetric):
    def evaluate_trace(self, trace, **kwargs):
        if trace.attributes.get("ai.trace.type") != "llm-invocation":
            return

        input_tokens = trace.attributes["ai.llm.response.usage"]["inputTokens"]
        output_tokens = trace.attributes["ai.llm.response.usage"]["outputTokens"]

        return [
            Measurement(
                name="TotalTokens",
                value=input_tokens + output_tokens,
                unit=Unit.Count,
            ),
            Measurement(
                name="InputTokens",
                value=input_tokens,
                unit=Unit.Count,
            ),
            Measurement(
                name="OutputTokens",
                value=output_tokens,
                unit=Unit.Count,
                validation_passed=output_tokens <= 100,  # added, just an example
            ),
        ]
```

And run evaluation again:

```python
results = GenerativeAIToolkit.eval(
    metrics=[TokenCount()],
    traces=[agent.traces]
)
results.summary()
```

Would now e.g. print (note `Total Nr Passed` and `Total Nr Failed`):

```
+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
| Avg NrOfOInputTokens | Avg NrOfOutputTokens | Avg Trace count per run | Avg LLM calls per run | Avg Tool calls per run | Total Nr Passed | Total Nr Failed |
+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
|        371.5         |         31.0         |           3.0           |          2.0          |          1.0           |        1        |        1        |
+----------------------+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
```

#### Additional information

You can attach additional information to the measurements you create. This information will be visible in the Web UI for conversation debugging, as well as in Amazon CloudWatch (if you use the seamless export of the measurements to CloudWatch, see further below):

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class MyMetric(BaseMetric):
    def evaluate_trace(self, trace, **kwargs):
        return Measurement(
            name="MyMeasurementName",
            value=123.456,
            unit=Unit.Count,
            additional_information={
                "context": "This is some context",
                "you": ["can store", "anything", "here"]
            }
        )
```

### 2.5 Repeatable Cases

You can create repeatable cases to run against your LLM application. The process is this:

```mermaid
flowchart LR
    A["Create LLM application (agent)"]
    B[Creates cases]
    C["Generate traces by running the cases against the LLM application (agent)"]
    D[Evaluate the traces with metrics]
    A --> B --> C --> D
```

A case has a name (optional) and user inputs. Each user input will be fed to the agent sequentially in the same conversation:

```python
my_case = Case(
    name="User wants to do something fun",
    user_inputs=[
        "I wanna go somewhere fun",
        "Within 60 minutes",
        "A museum of modern art",
    ],
)
```

A case can be run against an agent like this, returning the traces collected:

```python
traces = my_case.run(agent)
```

That will play out the conversation, feeding each input to the agent, awaiting its response, and then feeding the nextm until all user inputs have been fed to the agent. For quick tests this works, but if you have many cases you'll want to use `generate_traces()` (see below) to run them parallelized in bulk.

#### Cases with expectations

Here is a case with overall expectations, that will be interpreted by the `ConversationExpectationMetric` (if you include that metric upon calling `GenerativeAIToolkit.eval()` against the collected traces):

```python
import textwrap


conv_expectation_case = Case(
    name="User wants to go MoMA",
    user_inputs=[
        "I wanna go somewhere fun",
        "Within 60 minutes",
        "A museum of modern art",
    ],
    overall_expectations=textwrap.dedent(
        """
        The agent first asks the user (1) what type of activity they want to do and (2) how long they're wiling to drive to get there.
        When the user only answers the time question (2), the agent asks the user again what type of activity they want to do (1).
        Then, when the user finally answers the wat question also (1), the agent makes some relevant recommendations, and asks the user to pick.
        """
    ),
)
```

Here is a case with expectations per turn, that will be interpreted by the `AgentResponseSimilarityMetric` and `BleuMetric` (if you include any of these metrics upon calling `GenerativeAIToolkit.eval()` against the collected traces):

```python
similarity_case = Case(
    name="User wants to go to a museum",
)
similarity_case.add_turn(
    "I want to do something fun",
    [
        "To help you I need more information. What type of activity do you want to do and how long are you willing to drive to get there?",
        "Okay, to find some fun activities for you, I'll need a bit more information first. What kind of things are you interested in doing? Are you looking for outdoor activities, cultural attractions, dining, or something else? And how much time are you willing to spend driving to get there?",
    ],
)
similarity_case.add_turn(
    "I'm thinking of going to a museum",
    [
        "How long are you willing to drive to get there?"
        "Got it, you're interested in visiting a museum. That's helpful to know. What's the maximum amount of time you're willing to drive to get to the museum?"
    ],
)
```

#### Cases with dynamic input

Instead of listing out all user inputs beforehand, you can provide a user input producer to a case, which is a python function that dynamically creates user inputs to match the conversation. This can be of use during development, to e.g. do smoke tests to get a sense for how well the agent works.

The `user_input_producer` should be passed to the `Case` and it must be a Python `Callable` that accepts the parameter `messages`, which contains the conversation history. The `user_input_producer` should return new user input each time it's called, or an empty string to signal the conversation should end.

You can create your own user input producer implementation, or use the out-of-the-box `UserInputProducer` that uses an LLM under the hood to determine the next user utterance:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Case, UserInputProducer

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    system_prompt="You help users with movie suggestions. You are succinct and to-the-point"
)

def get_movie_suggestion(genre: str):
    """
    Generates a random movie suggestion, for the provided genre.
    Returns one movie suggestion (title) without any further information.
    Ensure the user provides a genre, do not assume the genre––ask the user if not provided.


    Parameters
    ----------
    genre : str
        The genre of the movie to be suggested.
    """
    return "The alleyways of Amsterdam (1996)"

agent.register_tool(get_movie_suggestion)

# This case does not have user inputs, but rather a user_input_producer,
# in this case the UserInputProducer class, which should be instantiated with the user's intent:
case = Case(name="User wants a movie suggestion", user_input_producer=UserInputProducer(user_intent="User wants a movie suggestion"))

traces = case.run(agent)

for trace in traces:
    print(trace.as_human_readable())
```

Would print e.g.:

```
[120027b89023dd54f59c50499b57b599/root/9e22ad550295191f] BedrockConverseAgent SERVER 2025-04-16T09:01:10.466Z - converse (ai.trace.type='converse' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
       Input: I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?
    Response: I can provide movie suggestions for different genres. What genre would you like a recommendation for? Examples of genres are action, comedy, drama, romance, horror, sci-fi, etc.

[120027b89023dd54f59c50499b57b599/9e22ad550295191f/1b2b94552c644558] BedrockConverseAgent CLIENT 2025-04-16T09:01:10.467Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'user', 'content': [{'text': "I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?"}]}

[120027b89023dd54f59c50499b57b599/9e22ad550295191f/19c00ed498f0abee] BedrockConverseAgent CLIENT 2025-04-16T09:01:10.467Z - conversation-history-list (ai.trace.type='conversation-history-list' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
    Messages: [{'role': 'user', 'content': [{'text': "I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?"}]}]

[120027b89023dd54f59c50499b57b599/9e22ad550295191f/fece03d8bc85f4d8] BedrockConverseAgent CLIENT 2025-04-16T09:01:10.467Z - llm-invocation (ai.trace.type='llm-invocation' peer.service='llm:claude-3-sonnet' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
Last message: [{'text': "I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?"}]
    Response: {'message': {'role': 'assistant', 'content': [{'text': 'I can provide movie suggestions for different genres. What genre would you like a recommendation for? Examples of genres are action, comedy, dra
              ma, romance, horror, sci-fi, etc.'}]}}

[120027b89023dd54f59c50499b57b599/9e22ad550295191f/00bf227f03f6f4ae] BedrockConverseAgent CLIENT 2025-04-16T09:01:11.613Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'assistant', 'content': [{'text': 'I can provide movie suggestions for different genres. What genre would you like a recommendation for? Examples of genres are action, comedy, drama, romance,
              horror, sci-fi, etc.'}]}

[3fdef11a72df06eb74fba3d65402d0da/root/f6a5671219710173] BedrockConverseAgent SERVER 2025-04-16T09:01:12.835Z - converse (ai.trace.type='converse' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
       Input: I'm interested in comedies. Do you have any good comedy movie suggestions?
    Response: The comedy movie suggestion is "The Alleyways of Amsterdam" from 1996. It sounds like an offbeat, quirky comedy set in the Netherlands. Let me know if you'd like another comedy recommendation or if th
              at piqued your interest!

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/fe6efe3b5c21310d] BedrockConverseAgent CLIENT 2025-04-16T09:01:12.835Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'user', 'content': [{'text': "I'm interested in comedies. Do you have any good comedy movie suggestions?"}]}

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/6a1566cb25737d02] BedrockConverseAgent CLIENT 2025-04-16T09:01:12.835Z - conversation-history-list (ai.trace.type='conversation-history-list' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
    Messages: [{'role': 'user', 'content': [{'text': "I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?"}]}, {'role': 'assistant', 'content': [{'text': 'I can provi
              de movie suggestions for different genres. What genre would you like a recommendation for? Examples of genres are action, comedy, drama, romance, horror, sci-fi, etc.'}]}, {'role': 'user', 'content':
              [{'text': "I'm interested in comedies. Do you have any good comedy movie suggestions?"}]}]

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/4723b2cb16046645] BedrockConverseAgent CLIENT 2025-04-16T09:01:12.835Z - llm-invocation (ai.trace.type='llm-invocation' peer.service='llm:claude-3-sonnet' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
Last message: [{'text': "I'm interested in comedies. Do you have any good comedy movie suggestions?"}]
    Response: {'message': {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'name': 'get_movie_suggestion', 'input': {'genre': 'comedy'}}}]}}

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/b49ca36d024d0150] BedrockConverseAgent CLIENT 2025-04-16T09:01:13.832Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'name': 'get_movie_suggestion', 'input': {'genre': 'comedy'}}}]}

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/bcc78bb3c1c1a110] BedrockConverseAgent CLIENT 2025-04-16T09:01:13.832Z - get_movie_suggestion (ai.trace.type='tool-invocation' peer.service='tool:get_movie_suggestion' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
       Input: {'genre': 'comedy'}
      Output: The alleyways of Amsterdam (1996)

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/c89ae710cadf9952] BedrockConverseAgent CLIENT 2025-04-16T09:01:13.832Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'user', 'content': [{'toolResult': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'status': 'success', 'content': [{'json': {'toolResponse': 'The alleyways of Amsterdam (1996)'}}]}}]}

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/a83184f18d57e2e0] BedrockConverseAgent CLIENT 2025-04-16T09:01:13.832Z - conversation-history-list (ai.trace.type='conversation-history-list' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
    Messages: [{'role': 'user', 'content': [{'text': "I'd like to get a movie recommendation. What genres or types of movies do you have suggestions for?"}]}, {'role': 'assistant', 'content': [{'text': 'I can provi
              de movie suggestions for different genres. What genre would you like a recommendation for? Examples of genres are action, comedy, drama, romance, horror, sci-fi, etc.'}]}, {'role': 'user', 'content':
              [{'text': "I'm interested in comedies. Do you have any good comedy movie suggestions?"}]}, {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'name': 'get_mo
              vie_suggestion', 'input': {'genre': 'comedy'}}}]}, {'role': 'user', 'content': [{'toolResult': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'status': 'success', 'content': [{'json': {'toolRespon...

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/b6a64ab8eb8f4cfc] BedrockConverseAgent CLIENT 2025-04-16T09:01:13.832Z - llm-invocation (ai.trace.type='llm-invocation' peer.service='llm:claude-3-sonnet' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
Last message: [{'toolResult': {'toolUseId': 'tooluse_Tf776MWLQ_iIyuYGsdvvTw', 'status': 'success', 'content': [{'json': {'toolResponse': 'The alleyways of Amsterdam (1996)'}}]}}]
    Response: {'message': {'role': 'assistant', 'content': [{'text': 'The comedy movie suggestion is "The Alleyways of Amsterdam" from 1996. It sounds like an offbeat, quirky comedy set in the Netherlands. Let me k
              now if you\'d like another comedy recommendation or if that piqued your interest!'}]}}

[3fdef11a72df06eb74fba3d65402d0da/f6a5671219710173/1ade2dbc33d3db6a] BedrockConverseAgent CLIENT 2025-04-16T09:01:15.410Z - conversation-history-add (ai.trace.type='conversation-history-add' peer.service='memory:short-term' ai.conversation.id='01JRYX9AZGZQNYJTQ4V4T3SCGJ' ai.auth.context='null')
     Message: {'role': 'assistant', 'content': [{'text': 'The comedy movie suggestion is "The Alleyways of Amsterdam" from 1996. It sounds like an offbeat, quirky comedy set in the Netherlands. Let me know if you\'
              d like another comedy recommendation or if that piqued your interest!'}]}
```

What you can see is that the agent asked the user a question because it needed more information (the genre, see first `SERVER` trace), and the user input producer provided an answer on behalf of the user: comedy (see second `SERVER` trace).

Note that you can still provide `user_inputs` in the case as well: these will be played out first, and once these are exhausted the `user_input_producer` will be invoked for getting subsequent user inputs. This way, you can 'prime' a conversation.

### 2.6 Cases with dynamic expectations

Cases can also be validated by passing it one or more validator functions. A validator function must be a Python `Callable` that accepts as input the traces of the conversation. Based on these traces the validator function should return `None` or an empty string, if the test passes. If the test fails it should return one or more messages (`str` or `Sequence[str]`).

The validator function will be invoked when the traces of the case are ran through `GenerativeAIToolkit.eval()` and this will generate measurements automatically: measurements with name `ValidationPassed` if the test passed (i.e. it returned `None` or `""`) and `ValidationFailed` otherwise. If the validation failed, the message that was returned will be included in the measurement's `additional_info` (or if an exception was thrown, the exception message):

```python
def validate_weather_report(traces: Sequence[CaseTrace]):
    root_trace = traces[0]
    last_output = root_trace.attributes["ai.agent.response"]
    if last_output.startswith("The weather will be"):
        # Test passed!
        return
    return f"Unexpected message: {last_output}"


case1 = Case(
    name="Check weather",
    user_inputs=["What is the weather like right now?"],
    validate=validate_weather_report,
)

traces = case1.run(agent)

# To run the validator functions, run GenerativeAIToolkit.eval()
# Validator functions will be run always, even if no metrics are provided otherwise:
results = GenerativeAIToolkit.eval(metrics=[], traces=[traces])

results.summary()

for conversation_measurements in results:
    for measurement in conversation_measurements.measurements:
        print(measurement)
```

That would e.g. print one failure (if the case has at least one failed validation, it is counted as a failure) and corresponding measurement:

```
+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
| Avg ValidationFailed | Avg Trace count per run | Avg LLM calls per run | Avg Tool calls per run | Total Nr Passed | Total Nr Failed |
+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+
|         1.0          |           3.0           |          2.0          |          1.0           |        0        |        1        |
+----------------------+-------------------------+-----------------------+------------------------+-----------------+-----------------+

Measurement(name='ValidationFailed', value=1, unit=<Unit.None_: 'None'>, additional_info={'validation_messages': ['Unexpected message: The current weather is sunny. Let me know if you need any other weather details!']}, dimensions=[], validation_passed=False)
```

### 2.7 Generating traces: running cases in bulk

When you have many cases, instead of calling `case.run(agent)` for each case, it's better to run cases in parallel like so:

```python
from generative_ai_toolkit.evaluate.interactive import GenerativeAIToolkit, Permute


traces = GenerativeAIToolkit.generate_traces(
    cases=cases, # pass in an array of cases here
    nr_runs_per_case=3, # nr of times to run each case, to account for LLM indeterminism
    agent_factory=BedrockConverseAgent, # This can also be your own factory function
    agent_parameters={
        "system_prompt": Permute(
            [
                "You are a helpful assistant",
                "You are a lazy assistant who prefers to joke around rather than to help users",
            ]
        ),
        "temperature": 0.0,
        "tools": my_tools, # list of python functions that can be used as tools
        "model_id": Permute(
            [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ]
        ),
    },
)
```

Explanation:

- `generate_traces()` is in essence nothing but a parallelized (with threads) invocation of `case.run(agent)` for each case provided. To account for LLM indeterminism, each case is run `nr_runs_per_case` times.
- Because an agent instantiation can only handle one conversation at a time, you must pass an `agent_factory` to `generate_traces()` so that it can create a fresh agent instance for each test conversation that it will run through. The `agent_factory` must be a python callable that can be fed `agent_parameters` and returns an agent instance. This can be a `BedrockConverseAgent` as above, but may be any Python object that exposes a `converse` method and `traces` property.
- The (optional) `agent_parameters` will be supplied to the `agent_factory` you provided.
- By using `Permute` for values within the `agent_parameters` you can test different parameter values against each other. In the example above, 2 different system prompts are tried, and 2 different model ID's. This in effect means 4 permutations (2 x 2) will be tried, i.e. the full cartesian product.
- The overall number of conversations that will be run is: `len(cases) * nr_runs_per_case * len(permutations)`

The return value of `generate_traces()` is an iterable of conversations, where each conversation is an array of traces. This makes sense because `case.run(agent)` returns an array of traces, and `generate_traces()` can be thought of as simply running multiple instances of `case.run(agent)`.

The iterable can be fed directly to `GenerativeAIToolkit.eval()` (that was explained above):

```python
results = GenerativeAIToolkit.eval(
    metrics=your_list_of_metrics,
    traces=traces, # the iterable of conversations as returned by generate_traces()
)

results.summary() # this prints a table with averages to stdout
```

### 2.8 CloudWatch Custom Metrics

Measurements can be logged to CloudWatch Logs in [Embedded Metric Format (EMF)](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format.html) easily, to generate custom metrics within Amazon CloudWatch Metrics:

```python
from generative_ai_toolkit.evaluate import GenerativeAIToolkit
from generative_ai_toolkit.utils.logging import logger

traces = [...] # Generate traces
metrics = [...] # Define metrics

results = GenerativeAIToolkit.eval(
    traces=traces,
    metrics=metrics,
)
for conversation_measurements in results:
    # Emit EMF logs for measurements at conversation level:
    last_trace = conversation_measurements.traces[0].trace
    for measurement in conversation_measurements.measurements:
        logger.metric(
            measurement,
            conversation_id=conversation_measurements.conversation_id,
            auth_context=last_trace.attributes.get("auth_context"),
            additional_info=measurement.additional_info,
            namespace="GenerativeAIToolkit",
            common_dimensions={
                "MyCommonDimension": "MyDimensionValue"
            },
            timestamp=int(last_trace.started_at.timestamp() * 1000),
        )
    # Emit EMF logs for measurements at trace level:
    for conversation_traces in conversation_measurements.traces:
        trace = conversation_traces.trace
        for measurement in conversation_traces.measurements:
            logger.metric(
                measurement,
                conversation_id=conversation_measurements.conversation_id,
                auth_context=trace.attributes.get("auth_context"),
                trace_id=trace.trace_id,
                additional_info=measurement.additional_info,
                namespace="GenerativeAIToolkit",
                common_dimensions={
                    "MyCommonDimension": "MyDimensionValue"
                },
                timestamp=int(trace.started_at.timestamp() * 1000),
            )
```

> Note: the above is exactly what happens for you if you use the `generative_ai_toolkit.run.evaluate.AWSLambdaRunner`, see below.

> Note: if you run the above in AWS Lambda, the custom metrics will now be generated, because AWS Lambda writes to Amazon CloudWatch Logs automatically. Elsewhere, you would still need to send the lines from `stdout` to Amazon CloudWatch Logs.

After that, you can view the metrics in Amazon CloudWatch metrics, and you have the full functionality of Amazon CloudWatch Metrics at your disposal to graph these metrics, create alarms (e.g. based on threshold or anomaly), put on dashboards, etc:

<img src="./assets/images/sample-metric.png" alt="Sample Amazon Cloud Metric" width="1200" />

#### Using the `AWSLambdaRunner`

Here's an example AWS Lambda function implementation that uses the `AWSLambdaRunner` to run evaluatations and emit metrics as EMF logs. You should attach this Lambda function to a DynamoDB stream attached to a table with traces (i.e. one that gets populated by the `DynamoDBTracer`):

```python
from generative_ai_toolkit.metrics.modules.latency import LatencyMetric
from generative_ai_toolkit.run.evaluate import AWSLambdaRunner

metrics = [
    LatencyMetric(),
]

AWSLambdaRunner.configure(metrics=metrics, agent_name="MyAgent")
```

In your Lambda function definition, if the above file is stored as `index.py`, you would use `index.AWSLambdaRunner` as handler.

### 2.9 Deploying and Invoking the `BedrockConverseAgent`

> Also see our **sample notebook [deploying_on_aws.ipynb](/examples/deploying_on_aws.ipynb)**.

A typical deployment of an agent using the Generative AI Toolkit would be, per the [reference architecture](#reference-architecture) mentioned above:

1. An AWS Lambda Function that is exposed as Function URL, so that you can use HTTP to send user input to the agent, and get a streaming response back. The Function URL would e.g. accept POST requests with the user input passed as body: `{"user_input": "What is the capital of France?"}`. If a conversation is to be continued, you could e.g. pass its ID in HTTP header `x-conversation-id`. Correspondingly, when a new conversation is started, its ID would e.g. be passed back in the `x-conversation-id` response header. You can use the `Runner` from Generative AI Toolkit, to implement the Lambda function exactly like this, see below.
2. An Amazon DynamoDB table to store conversation history and traces. This table has a stream enabled. The AWS Lambda function, your agent, would write its traces to this table. Additionally (using the `TeeTracer` and the `OtlpTracer`) the agent would send the traces to AWS X-Ray for developer inspection.
3. An AWS Lambda Function, that is attached to the DynamoDB table stream, to run `GenerativeAIToolkit.eval()` on the collected traces. This Lambda function would write the collected measurements to stdout in EMF format (see above), to make the measurements available in Amazon CloudWatch Metrics.

#### Using the `Runner` to run your agent as Lambda function

The following code shows how you can implement your Generative AI Toolkit based agent as Lambda function, per the description above.

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.run.agent import Runner
from generative_ai_toolkit.tracer import TeeTracer
from generative_ai_toolkit.tracer.otlp import OtlpTracer
from generative_ai_toolkit.tracer.dynamodb import DynamoDbTracer


class MyAgent(BedrockConverseAgent):
    def __init__(self):
        super().__init__(
            model_id="eu.anthropic.claude-3-sonnet-20240229-v1:0",
            system_prompt="You are a helpful assistant",
            conversation_history=DynamoDbConversationHistory(
                table_name="messages"
            ),
            tracer=TeeTracer()
            .add_tracer(DynamoDbTracer(table_name="traces"))
            .add_tracer(OtlpTracer()),
        )


Runner.configure(
    agent=MyAgent,  # Agent factory
)
```

In your Lambda function definition, if the above file is stored as `index.py`, you would use `index.Runner()` as handler.

Note that you must use the [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) to run the `Runner` on AWS Lambda.

#### Invoking the AWS Lambda Function URL with the `IamAuthInvoker`

If you use the `Runner` just explained, you would deploy your agent as an AWS Lambda Function that is exposed as Function URL. You should enable IAM Auth, in which case you must [sign all requests with AWS Signature V4 as explained here](https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html).

This library has helper code on board to make that more easy for you. You can simply call a function and pass the user input. The response stream is returned as a Python iterator:

```python
from generative_ai_toolkit.utils.lambda_url import IamAuthInvoker

lambda_url_invoker = IamAuthInvoker(lambda_function_url="https://...")  # Pass your AWS Lambda Function URL here

response1 = lambda_url_invoker.converse_stream(
    user_input="What is the capital of France?"
)  # This returns an iterator that yields chunks of tokens

print("Conversation ID:", response1.conversation_id)

print()
for tokens in response1:
    print(tokens, end="", flush=True)

response2 = lambda_url_invoker.converse_stream(
    user_input="What are some touristic highlights there?",
    conversation_id=response1.conversation_id,  # continue conversation
)

print()
for tokens in response2:
    print(tokens, end="", flush=True)
```

#### Invoking the AWS Lambda Function URL with `curl`

Using `curl` works too because `curl` supports SigV4 out of the box:

```shell
curl -v \
  https://your-lambda-function-url \
  --data '{"user_input": "What is the capital of France?"}' \
  --header "x-conversation-id: $CONVERSATION_ID" \
  --header "Content-Type: application/json" \
  --header "x-amz-security-token: $AWS_SESSION_TOKEN" \
  --no-buffer \
  --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
  --aws-sigv4 "aws:amz:$AWS_REGION:lambda"
```

#### Deployments outside AWS Lambda e.g. containerized as a pod on EKS

The `Runner` is a WSGI application and can be run with any compatible server, such as `gunicorn`.

To support concurrency, make sure you pass an Agent factory to the Runner configuration, and not an Agent instance. Agent instances do not support concurrency and are meant to handle one request at a time. The same is true for Conversation History and Tracers, pass a factory to your agent, not an instance.

For example:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.run.agent import Runner


class MyAgent(BedrockConverseAgent):
    def __init__(self):
        super().__init__(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            system_prompt="You are a helpful assistant",
            conversation_history=DynamoDbConversationHistory(
                table_name="messages"
            ),
        )


Runner.configure(
    agent=MyAgent,  # Agent factory
    auth_context_fn=lambda _: "TestUser",  # Add your own implementation here! See "Security" section below.
)
```

If the above file would be e.g. on path `/path/to/agent.py`, then you can run it with `gunicorn` like so:

```shell
gunicorn "path.to.agent:Runner()"
```

Make sure to tune concurrency. By default `gunicorn` runs with 1 worker (process) and 1 thread. To e.g. support 20 concurrent conversations, you could run with 4 workers and 5 threads per worker:

```shell
gunicorn --workers 4 --threads 5 "path.to.agent:Runner()"
```

#### Security: ensuring users access their own conversation history only

You must make sure that users can only set the conversation ID to an ID of one of their own conversations, or they would be able to read conversations from other users (unless you want that of course). To make this work securely with the out-of-the-box `DynamoDbConversationHistory`, you need to set the right auth context on the agent for each conversation with a user.

Setting the auth context ensures that each conversation is bound to that auth context. Even if two users would (accidentally or maliciously) use the same conversation ID, the auth context would still limit each user to see his/her own conversations only. This works because the auth context is part of the Amazon DynamoDB key.

In the simplest case, you would use the user ID as auth context. For example, if you're using Amazon Cognito, you could use the `sub` claim from the user's access token as auth context.

You can manually set the auth context on a `BedrockConverseAgent` instance like so (and this is propagated to the conversation history instance your agent uses):

```python
agent.set_auth_context("<my-user-id>")
```

> If you use the `Runner` (see above) you don't have to call `agent.set_auth_context(...)` manually, but rather you should provide an `auth_context_fn`, which is explained in the next paragraph.

> If you have custom needs, for example you want to allow some users, but not all, to share conversations, you likely need to implement a custom conversation history class to support your auth context scheme (e.g. you could subclass `DynamoDbConversationHistory` and customize the logic).

The deployment of the `BedrockConverseAgent` with AWS Lambda Function URL, explained above, presumes you're wrapping this component inside your architecture in some way, so that it is not actually directly invoked by users (i.e. real users don't use `curl` to invoke the agent as in the example above) but rather by another component in your architecture. As example, let's say you're implementing an architecture where the user's client (say an iOS app) connects to a backend-for-frontend API, that is responsible, amongst other things, for ensuring users are properly authenticated. The backend-for-frontend API may then invoke the `BedrockConverseAgent` via the AWS Lambda function URL, passing the (verified) user ID in the HTTP header `x-user-id`:

```mermaid
flowchart LR
    A[User]
    B[iOS app]
    C["Backend-for-frontend"]
    D["BedrockConverseAgent exposed via AWS Lambda function URL"]
    A --> B --> C --> D
```

In this case, configure the `Runner` (from `generative_ai_toolkit.run.agent`) to use the incoming HTTP header `x-user-id` as auth context:

```python
from flask import Request
from generative_ai_toolkit.run.agent import Runner

def extract_x_user_id_from_request(request: Request):
    return request.headers["x-user-id"] # Make sure you can trust this header value!

Runner.configure(agent=my_agent, auth_context_fn=extract_x_user_id_from_request)
```

> The `Runner` uses, by default, the AWS IAM `userId` as auth context. The actual value of this `userId` depends on how you've acquired AWS credentials to sign the AWS Lambda Function URL request with. For example, if you've assumed an AWS IAM Role it will simply be the concatenation of your assumed role ID with your chosen session ID. You'll likely want to customize the auth context as explained in this paragraph!

### 2.10 Web UI for Conversation Debugging

The Generative AI Toolkit provides a local, web-based user interface (UI) to help you inspect and debug conversations, view evaluation results, and analyze agent behavior. This UI is particularly useful during development and testing phases, allowing you to quickly identify issues, review traces, and understand how your agent processes user queries and responds.

**Key Features:**

- **Trace Inspection:** View the entire sequence of interactions, including user messages, agent responses, and tool invocations. Traces are displayed in chronological order, accompanied by detailed metadata (timestamps, token counts, latencies, costs), making it easier to pinpoint performance bottlenecks or unexpected behaviors.

- **Conversation Overview:** Each conversation is presented as a cohesive flow. You can navigate through every turn in a conversation to see how the context evolves over time, how the agent utilizes tools, and how different system prompts or model parameters influence the responses.

- **Metrics and Evaluation Results:** When you run `GenerativeAIToolkit.eval()` on the collected traces, the UI provides a clear visualization of the results. This includes SQL query accuracy metrics, cost estimates, latency measurements, and custom validation checks. The UI helps you identify which cases passed or failed, and the reasons why.

Below are two example screenshots of the UI in action:

_In this screenshot, you can see multiple conversations along with their metrics and pass/fail status. Clicking the View button for a conversation reveals its detailed traces and metrics:_

<img src="./assets/images/ui-measurements-overview.png" alt="UI Measurements Overview Screenshot" title="UI Measurements Overview" width="1200"/>

_Here, a single conversation’s full trace is displayed. You can see user queries, agent responses, any tool calls made, and evaluation details like latency and cost. This view helps you understand how and why the agent produced its final answer:_

<img src="./assets/images/ui-conversation.png" alt="UI Conversation Display Screenshot" title="UI Conversation Display" width="1200"/>

**How to Launch the UI:**

After generating and evaluating traces, start the UI by calling:

```python
results.ui.launch()
```

This command runs a local web server (at http://localhost:7860) where you can interact with the web UI. When you have finished inspecting your conversations and metrics, you can shut down the UI by running:

```python
results.ui.close()
```

### 2.11 Mocking and Testing

As with all software, you'll want to test your agent. You can use above mentioned [Cases](#25-repeatable-cases) for evaluating your agent in an end-to-end testing style. You may also want to create integration tests and unit tests, e.g. to target specific code paths in isolation. For such tests you can use the following tools from the Generative AI Toolkit:

- The **`MockBedrockConverse`** class allows you to **mock** the Bedrock Converse API in a conversational way, so that you can steer your agent towards particular actions that you want to test. The LLM is the brain of the agent, so if you control the brain, you control the agent.
- The **`Expect`** class allows you to express your assertions on test executions in a concise way. With the `Expect` class you write assertions against the collected traces, so you can test your agent on a deep level. You can write assertions against everything the agent traces, e.g. all tool invocations, LLM invocations, access of conversational memory, user input, agent response, etc.

Let's see both in action. Here's a sample agent that we'll test. Note that we're instantiating it with the Bedrock Converse API mock:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Expect, Case
from generative_ai_toolkit.test.mock import MockBedrockConverse

# create mock:
mock = MockBedrockConverse()

agent = BedrockConverseAgent(
    model_id="amazon.nova-lite-v1:0",
    session=mock.session(),  # use mock
)

def weather_tool(city: str, unit: str = "celsius") -> str:
    """
    Get the weather report for a city

    Parameters
    ---
    city: str
      The city
    unit: str
      The unit of degrees (e.g. celsius)
    """
    return f"The weather in {city} is 20 degrees {unit}."

agent.register_tool(weather_tool)
```

#### My first unit test

Now, to write a test, we load mock responses into our mock. When the agent then invokes the Amazon Bedrock Converse API, it will actually invoke our mock instead, and thus use the responses we prepared:

```python
# prepare mock response:
sample_response = "Hello, how can I help you today"
mock.add_output(text_output=[sample_response])

# invoke agent:
response = agent.converse("Hi there!")

# assert the agent's response matches our expectation:
assert response == sample_response

# equivalent to the assert statement:
Expect(agent.traces).agent_text_response.to_equal(sample_response)
```

#### Preparing a sequence of responses

The following example shows how the mock responses are played out in sequence:

```python
# reset agent and mock:
agent.reset()
mock.reset()

# prepare mock responses:
sample_response1 = "Hello, how can I help you today"
mock.add_output(text_output=[sample_response1])
sample_response2 = "I don't have a name"
mock.add_output(text_output=[sample_response2])

# run conversation through:
Case(["Hi there!", "What's your name?"]).run(agent)

# check agent responses:
Expect(agent.traces).agent_text_response.at(0).to_equal(sample_response1)
Expect(agent.traces).agent_text_response.to_equal(sample_response2)
```

#### Testing tool invocations

It becomes more interesting if you want to test tool invocations. The sequence under the hood may then be:

1. Agent invokes LLM --> LLM tells it to invoke a tool
2. Agent invokes tool
3. Agent invokes LLM with tool results --> LLM tells it to return a response to the user

Here's how to test that. Notice how the `Expect` class allows you to test the inner workings of the agent, e.g. the tool invocations:

```python
# reset agent and mock:
agent.reset()
mock.reset()

# prepare mock responses, including a tool use:
mock.add_output(
    text_output=["Okay, let me check the weather for you."],
    tool_use_output=[{"name": "weather_tool", "input": {"city": "Amsterdam"}}],
)
mock.add_output(text_output=["It's nice and sunny in Amsterdam!"])

# run conversation through:
Case(["Hi there! What's the weather like in Amsterdam?"]).run(agent)

# check agent responses, and tool invocations:
Expect(agent.traces).user_input.to_include("What's the weather like in Amsterdam?")
Expect(agent.traces).tool_invocations.to_have_length()
Expect(agent.traces).tool_invocations.to_include("weather_tool").with_input(
    {"city": "Amsterdam"}
).with_output("The weather in Amsterdam is 20 degrees celsius.")
Expect(agent.traces).agent_text_response.to_equal(
    "Okay, let me check the weather for you.\nIt's nice and sunny in Amsterdam!"
)
```

#### Mixing mock and real responses

You can also mix mock reponses and real response. E.g. you may want to 'prime' a conversation by first using mock responses, and after that allow the agent to invoke the real Amazon Bedrock Converse API:

```python
# reset agent and mock:
agent.reset()
mock.reset()

# prepare mock responses, including a tool use:
mock.add_output(
    text_output=["Okay, let me check the weather for you."],
    tool_use_output=[{"name": "weather_tool", "input": {"city": "Amsterdam"}}],
)

# allow the agent to invoke Bedrock once:
mock.add_real_response()

# run conversation through:
Case(["Hi there! What's the weather like in Amsterdam?"]).run(agent)

# check agent responses, and tool invocations:
Expect(agent.traces).tool_invocations.to_have_length()
Expect(agent.traces).tool_invocations.to_include("weather_tool").with_input(
    {"city": "Amsterdam"}
).with_output("The weather in Amsterdam is 20 degrees celsius.")
Expect(agent.traces).user_input.to_include("What's the weather like in Amsterdam?")

# We have to be a bit more lenient with our assertion now,
# because the agent's response is not deterministic anymore!:
Expect(agent.traces).agent_text_response.to_include("20")
```

#### Usage with Pytest

Here's a Pytest example:

```python
import pytest

from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.test import Expect, Case
from generative_ai_toolkit.test.mock import MockBedrockConverse

@pytest.fixture
def mock_bedrock_converse():
    mock = MockBedrockConverse()
    yield mock
    if mock.mock_responses:
        raise Exception("Still have unconsumed mock responses")

@pytest.fixture
def my_agent(mock_bedrock_converse):
    agent = BedrockConverseAgent(
        model_id="amazon.nova-lite-v1:0",
        session=mock_bedrock_converse.session(),  # use mock
    )

    def weather_tool(city: str, unit: str = "celsius") -> str:
        """
        Get the weather report for a city

        Parameters
        ---
        city: str
          The city
        unit: str
          The unit of degrees (e.g. celsius)
        """
        return f"The weather in {city} is 20 degrees {unit}."

    agent.register_tool(weather_tool)
    yield agent

def test_agent(my_agent, mock_bedrock_converse):
    sample_response1 = "Hello, how can I help you today"
    mock_bedrock_converse.add_output(text_output=[sample_response1])
    sample_response2 = "I don't have a name"
    mock_bedrock_converse.add_output(text_output=[sample_response2])

    # run conversation through:
    Case(["Hi there!", "What's your name?"]).run(my_agent)

    # check agent responses:
    Expect(my_agent.traces).agent_text_response.at(0).to_equal(sample_response1)
    Expect(my_agent.traces).agent_text_response.to_equal(sample_response2)

```

Note that since we're using Pytest fixtures to provide a new mock and agent for each test case, we don't have to call `reset()` on them.

### 2.12 Model Context Protocol (MCP) Client

You can turn your agent into an MCP client easily like so:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.mcp.client import McpClient

# Create agent:
agent = BedrockConverseAgent(
    system_prompt="You are a helpful assistant",
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
)

# Turn agent into MCP client:
mcp_client = McpClient(agent)
```

When you instantiate the `McpClient` it will look for an MCP configuration (`mcp.json`) to load MCP servers. All MCP servers from the configuration will be added as tools to the agent automatically.

> **IMPORTANT**: The implementation does not ask for approval before using any tools. It is thus imperative that you only add trusted MCP servers to your configuration.

To load the configuration, `mcp.json` in the current working directory is tried first, and then `~/.aws/amazonq/mcp.json` (which is the Amazon Q config path). If both do not exist, no tools will be added to the agent.

You can also provide the path to `mcp.json` explicitly upon instantiating the McpClient:

```python
mcp_client = McpClient(agent, client_config_path="/path/to/mcp.json")
```

The `mcp.json` config follows the same format as Amazon Q MCP config, e.g.:

```json
{
  "mcpServers": {
    "WeatherForecasts": {
      "command": "python3",
      "args": ["mcp_server_get_weather.py"],
      "env": {
        "WEATHER": "Sunny",
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

Note: only local MCP servers (that communicate over `stdio`) are supported currently.

#### Chat loop

To chat with your MCP client, call `chat()`. This will start a REPL chat:

```python
mcp_client.chat()
```

```
MCP server configuration loaded: mcp.json

MCP client ready. Type /q to quit. Type /t to list the available tools.

You: /t

Listing available tools:

  current_weather
  _______________

    Gets the current weather for the user.

    This tool is already aware of the user's location, so you don't need to provide it.

You: How's the weather?
Assistant: It's currently sunny and 27 degrees Celsius outside.

You: /q
Assistant: Goodbye!
```

You can customize the chat loop by providing your own loop function:

```python
def my_chat_loop(agent: Agent):
    while True:
        user_input = input("Awesome user: ")
        if not user_input:
            break
        for chunk in agent.converse_stream(user_input):
            print(chunk, end="", flush=True)
        print()
```

And then:

```python
mcp_client.chat(chat_loop=my_chat_loop)
```

```
Awesome user:

```
