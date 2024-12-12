# Generative AI Toolkit

The **Generative AI Toolkit** is a lightweight library that covers the life cycle of LLM-based applications, including agents. Its purpose is to support developers in building and operating high quality LLM-based applications, over their entire life cycle, starting with the very first deployment, in an automated workflow.

The Generative AI Toolkit makes it easy to measure and test the performance of LLM-based applications, during development as well as in production. Measurements and test results can be integrated seamlessly with Amazon CloudWatch Metrics.

## Architecture

<img src="./assets/images/architecture.png" alt="Architecture" width="1200" />

## Key Terms

To fully utilize the Generative AI Toolkit, it’s essential to understand the following key terms:

- **Traces**: Traces are records of interactions between the user and the LLM or tools. They capture the entire request-response cycle, including input prompts, model outputs, tool calls, and metadata such as latency, token usage, and execution details. Traces form the foundation for evaluating an LLM's behavior and performance.

- **Metrics**: Metrics are measurements derived from traces that evaluate various aspects of an LLM's performance. Examples include latency, token usage, similarity with expected responses, sentiment, and cost. Metrics can be customized to measure specific behaviors or to enforce validation rules.

- **Cases**: Cases are repeatable tests that simulate conversations with the agent. They consist of a sequence of user inputs and expected agent behaviors or outcomes. Cases are used to validate the agent's responses against defined expectations, ensuring consistent performance across scenarios.

- **Agents**: An agent is an implementation of an LLM-based application that processes user inputs and generates responses. The toolkit provides a simple and extensible agent implementation with built-in support for tracing and tool integration.

- **Tools**: Tools are external functions or APIs that agents can invoke to provide additional capabilities (e.g., fetching weather data or querying a database). Tools are registered with agents and seamlessly integrated into the conversation flow.

- **Conversation History**: This refers to the sequence of messages exchanged between the user and the agent. It can be stored in memory or persisted to external storage, such as DynamoDB, to maintain context across sessions.

- **CloudWatch Custom Metrics**: These are metrics logged to Amazon CloudWatch in Embedded Metric Format (EMF), enabling the creation of dashboards, alarms, and aggregations to monitor agent performance in production environments.

- **Web UI**: A local web-based interface that allows developers to inspect traces, debug conversations, and view evaluation results interactively. This is particularly useful for identifying and resolving issues in the agent's responses.

## Getting started

To get started, you can follow either of these paths:

### Cookiecutter

You can bootstrap an AWS CDK project with a vanilla agent using the **cookiecutter template**. This is the easiest and quickest way to get to a working agent that runs locally as well as on AWS and can be invoked over HTTPS, e.g. with `curl`. The vanilla agent is meant both as an example and as a starting point to your own development: we recommend you play with the vanilla agent and the included notebooks with sample code and explanations. You can then proceed to customize the agent and the metrics to your liking. Proceed to [1.0 Cookiecutter template](#10-cookiecutter-template).

### Generative AI Toolkit Library

You can install the `generative_ai_toolkit` and explore how to create reliable agents with it in an IPython notebook or interactive Python shell. Proceed to [2.0 Generative AI Toolkit Agent](#20-generative-ai-toolkit-agent).

### Examples

In addition to the cookiecutter template and basic instructions, we are providing a collection of example notebooks in the `examples` folder.

For instance, if you’re interested in generating SQL queries from natural language inputs, refer to the **text-to-sql** notebook in `examples/text_to_sql.ipynb` which includes the Generative AI Toolkit Web UI:

<img src="./assets/images/generative-ai-toolkit-webui-detail.png" alt="UI Overview Screenshot" title="UI Overview Screenshot" width="1200"/>

These examples serve as practical guides and starting points for integrating the toolkit into your own workflows. Feel free to adapt and extend them to suit your specific application needs.


## 1.0 Cookiecutter template

The cookiecutter template in this repository will create a new Generative AI Toolkit project for you: an AWS CDK project with sample code and a notebook with explanations, with a vanilla agent onboard that you can customize to your liking.

#### Prereqs

- Install [cookiecutter](https://www.cookiecutter.io/), e.g. with: `pipx install cookiecutter`

#### Usage

Use the cookiecutter template to create the new project. Follow the prompts to create a new folder with everything you need to get started in it, amongst which a vanilla agent that you can deploy as-is or customize:

```shell
cookiecutter https://github.com/awslabs/generative-ai-toolkit
```

After this you can simply deploy your agent with `cdk deploy` and test it with the `test_function_url.sh` script. Follow the `README.md` in your new project folder for more details.

## 2.0 Generative AI Toolkit

#### Table of Contents

2.1 [Installation](#21-installation)  
2.2 [Agent Implementation](#22-agent-implementation)  
2.3 [Tracing](#23-tracing)  
2.4 [Metrics](#24-metrics)  
2.5 [Repeatable Cases](#25-repeatable-cases)  
2.6 [Cases with Dynamic Expectations](#26-cases-with-dynamic-expectations)  
2.7 [Generating Traces: Running Cases in Bulk](#27-generating-traces-running-cases-in-bulk)  
2.8 [CloudWatch Custom Metrics](#28-cloudwatch-custom-metrics)  
2.9 [Deploying and Invoking the BedrockConverseAgent](#29-deploying-and-invoking-the-bedrockconverseagent)  
2.10 [Web UI for Conversation Debugging](#210-web-ui-for-conversation-debugging)

### 2.1 Installation

Install `generative_ai_toolkit` with support for all features, amongst which interactive evaluation of metrics:

```bash
pip install "generative-ai-toolkit[all]"
```

### 2.2 Agent implementation

The heart of the Generative AI Toolkit are the traces it collects, that are the basis for evaluations (explained below). The toolkit includes a simple agent implementation that is backed by the [Amazon Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) and that is instrumented to collect traces in the right format.

A benefit of using this agent implementation, is that you can run the agent locally––it doesn't require any AWS deployment at all and only needs Amazon Bedrock model access. You can quickly iterate and try different agent settings, such as the backing LLM model id, system prompt, temperature, tools, etc. You can create repeatable test cases and run extensive and rigorous evaluations locally.

> We'll first explain how our agent implementation works. Feel free to directly skip to the explanation of [Tracing](#tracing) or [Metrics](#metrics) instead.

The Generative AI Toolkit Agent implementation is simple and lightweight, and makes for a no-nonsense developer experience. You can easily instantiate and converse with agents while working in the Python interpreter (REPL) or in a notebook:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
```

That's it. You now have an agent you can chat with.

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
print(len(agent.messages)) # 1
agent.reset()
print(len(agent.messages)) # 0
print(agent.conversation_id)  # e.g.: "01J5DQRD864TR3BF314CZK8X5B" (changed)
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

You can make `BedrockConverseAgent` log traces of the LLM and tool calls it performs, by providing a tracer class, such as the `InMemoryAgentTracer`, or the `DynamoDbAgentTracer` that logs traces to DynamoDB:

```python
from generative_ai_toolkit.agent import BedrockConverseAgent
from generative_ai_toolkit.conversation_history import DynamoDbConversationHistory
from generative_ai_toolkit.tracer import DynamoDbAgentTracer # Import tracer

agent = BedrockConverseAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    conversation_history=DynamoDbConversationHistory(table_name="conversations"),
    tracer=DynamoDbAgentTracer(table_name="traces"), # Add tracer, this table needs to exist, with string keys "pk" and "sk"
)
```

Now, when you `converse()` with the agent, and the agent calls the LLM and tools, it will log traces. You can inspect these traces in the AWS console or programmatically like so:

```python
agent.converse("What's the capital of France?")
print(agent.traces) # Prints the traces. In this example it would be just one trace of the LLM call
```

Traces can be of type `LLmTrace` or `ToolTrace`.

This is an example of an `LlmTrace`. As you can see it has the full detail of the call to the LLM, e.g. inputs, outputs, latency, nr of tokens, etc:

```python
LlmTrace(conversation_id='01J5DDQMC06ZEZKS5QPBAV4CYH', to='LLM', created_at=datetime.datetime(2024, 8, 16, 11, 4, 1, 152411, tzinfo=datetime.timezone.utc), request={'messages': [{'content': [{'text': "What's the capital of France?"}], 'role': 'user'}], 'system': [{'text': 'You are a helpful agent'}], 'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0', 'inferenceConfig': {}}, response={'output': {'message': {'content': [{'text': 'The capital of France is Paris.'}], 'role': 'assistant'}}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 409}, 'ResponseMetadata': {'HTTPHeaders': {'date': 'Fri, 16 Aug 2024 11:04:05 GMT', 'content-length': '212', 'content-type': 'application/json', 'connection': 'keep-alive', 'x-amzn-requestid': 'cb77b274-8786-447e-8446-e22a025adf0a'}, 'RequestId': 'cb77b274-8786-447e-8446-e22a025adf0a', 'HTTPStatusCode': 200, 'RetryAttempts': 0}, 'usage': {'outputTokens': 10, 'totalTokens': 29, 'inputTokens': 19}}, trace_id=Ulid('01J5DDQMC04C9J4SWGSK86HR3G'), additional_info={})
```

This is an example of a `ToolTrace`. As you can see it has the full detail of the call to the tool, e.g. inputs, outputs, latency, etc:

```python
ToolTrace(conversation_id='01J5DDQMC06ZEZKS5QPBAV4CYH', to='TOOL', request={'tool_name': 'bad_weather_report', 'tool_use_id': 'tooluse_pO8SQE5OT6O_VXy_P0XdDg', 'tool_input': {'city_name': 'Amsterdam'}}, response={'tool_response': {'tool_response': 'Rainy'}, 'latency_ms': 0}, created_at=datetime.datetime(2024, 8, 16, 11, 8, 35, 789444, tzinfo=datetime.timezone.utc), trace_id=Ulid('01J5DE00JDWDQQQ8TGQFJT61Z5'), additional_info={})
```

#### Viewing traces

Traces are represented (with Python's `repr()` call) minimally.

```python
repr(agent.traces)
```

Would e.g. print:

```
[Trace(to=LLM, conversation_id=01JD2CMHRJJ6BPW4DM7V9S7V3Q, trace_id=01JD2CMK7H30MJ11NG4GQDT1PD),
 Trace(to=TOOL, conversation_id=01JD2CMHRJJ6BPW4DM7V9S7V3Q, trace_id=01JD2CMK7HWZ7M6C1RPGJY4MNH),
 Trace(to=LLM, conversation_id=01JD2CMHRJJ6BPW4DM7V9S7V3Q, trace_id=01JD2CMM6P3H1ZHEKDS0TRXFR0)]
```

When transformed to a string, e.g. when printed, more information is displayed:

```python
print(agent.traces[0])
```

Would e.g. print:

```
======================================
LLM TRACE (LlmCaseTrace)
======================================
To:              LLM
Conversation ID: 01JD2CMHRJJ6BPW4DM7V9S7V3Q
Auth context:    None
Created at:      2024-11-21 08:22:15.546000+00:00
Additional info:
  {}
Request messages:
  {'text': 'What is the weather like right now?'}
Response message:
  [{'text': 'Okay, let me check the weather for your current location:'}, {'toolUse': {'toolUseId': 'tooluse_abox8YYlSeiPsjQWab0O5w', 'name': 'get_weather', 'input': {}}}]
Request (full):
  {"modelId":"anthropic.claude-3-haiku-20240307-v1:0", "inferenceConfig":{}, "messages":[{"role":"user", "content":[{"text":"What is the weather like right now?"}]}], "toolConfig":{"tools":[{"toolSpec":{"name":"get_weather", "description":"Gets the current weather for the user's location.\nThis tool has no parameters, and already knows where the user is.", "inputSchema":{"json":{"type":"object", "properties":{}}}}}]}}
Response (full):
  {"ResponseMetadata":{"RequestId":"9e4fc2a8-49e7-454e-8870-6a408bab27c3", "HTTPStatusCode":200, "HTTPHeaders":{"date":"Tue, 19 Nov 2024 14:18:48 GMT", "content-type":"application/json", "content-length":"332", "connection":"keep-alive", "x-amzn-requestid":"9e4fc2a8-49e7-454e-8870-6a408bab27c3"}, "RetryAttempts":0}, "output":{"message":{"role":"assistant", "content":[{"text":"Okay, let me check the weather for your current location:"}, {"toolUse":{"toolUseId":"tooluse_abox8YYlSeiPsjQWab0O5w", "name":"get_weather", "input":{}}}]}}, "stopReason":"tool_use", "usage":{"inputTokens":337, "outputTokens":50, "totalTokens":387}, "metrics":{"latencyMs":1379}}
```

Traces are Python `dataclasses` so you can also turn them into a `dict` and log them like that to see even fuller details:

```python
from dataclasses import asdict

print(asdict(agent.traces[0]))
```

### 2.4 Metrics

Metrics allow you to measure how well your agent performs. The Generative AI Toolkit comes with some metrics out of the box, and makes it easy to develop your own metric as well. Metrics work off of traces, and can measure anything that is represented within the traces.

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

Note that these measurements can easily be exported to Amazon CloudWatch as Custom Metrics, which allow you to use Amazon CloudWatch for creating dashboards, aggregations, alarms, etc. See further below.

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
            value=len([trace for trace in conversation_traces if trace.to == "TOOL"]),
            unit=Unit.Count,
        )
```

The above metric works at conversation level and therefore implements `evaluate_conversation` which gets all the traces from the conversation in one go.

Even more simple custom metrics would work at individual trace level, without needing to know about the other traces in the conversation. In that case, implement `evaluate_trace`:

> Note the `TokensMetric` actually comes out-of-the-box, but we'll reimplement it here for sake of the example

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class TokenCount(BaseMetric):
    def evaluate_trace(self, trace, **kwargs):
        if trace.to != "LLM":
            # Ignore tool traces
            return
        return [
            Measurement(
                name="NrOfOInputTokens",
                value=trace.response["usage"]["inputTokens"],
                unit=Unit.Count,
            ),
            Measurement(
                name="NrOfOutputTokens",
                value=trace.response["usage"]["outputTokens"],
                unit=Unit.Count,
            ),
        ]
```

The above custom metric returns 2 measurements, but only for LLM traces.

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

#### Passing or Failing a Custom Metric

Besides measuring an agent's performance in a scalar way, custom metrics can (optionally) return a Pass or Fail indicator. This will be reflected in the measurements summary and such traces would be marked as failed in the Web UI for conversation debugging (see further).

Let's tweak our `TokenCount` metric:

```python
from generative_ai_toolkit.metrics import BaseMetric, Measurement, Unit


class TokenCount(BaseMetric):
    def evaluate_trace(self, trace, **kwargs):
        if trace.to != "LLM":
            return
        return [
            Measurement(
                name="NrOfOInputTokens",
                value=trace.response["usage"]["inputTokens"],
                unit=Unit.Count,
            ),
            Measurement(
                name="NrOfOutputTokens",
                value=trace.response["usage"]["outputTokens"],
                unit=Unit.Count,
                validation_passed=trace.response["usage"]["outputTokens"] < 30,  # added, just an example
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

A case has a name and user inputs. Each user input will be fed to the agent sequentially in the same conversation:

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
    print(trace)
```

Would print e.g.:

```
======================================
LLM TRACE (LlmCaseTrace)
======================================
To:              LLM
Conversation ID: 01JD6X0YVVE97B0F8C0QXP8YT6
Auth context:    None
Created at:      2024-11-21 08:22:12.655000+00:00
Additional info:
  {}
Request messages:
  {'text': "I'd like to get a movie suggestion. What kind of movie would you recommend?"}
Response message:
  [{'text': "Okay, let's get a movie suggestion for you. What genre of movie would you like? I'll need that information to provide a relevant recommendation."}]
Request (full):
  ...
Response (full):
  ...
======================================
LLM TRACE (LlmCaseTrace)
======================================
To:              LLM
Conversation ID: 01JD6X0YVVE97B0F8C0QXP8YT6
Auth context:    None
Created at:      2024-11-21 08:22:15.546000+00:00
Additional info:
  {}
Request messages:
  {'text': "I'd like to see a comedy. Can you suggest a funny movie I might enjoy?"}
Response message:
  [{'toolUse': {'toolUseId': 'tooluse_hahjynWhQzi86Zpwzt4Ygw', 'name': 'get_movie_suggestion', 'input': {'genre': 'comedy'}}}]
Request (full):
  ...
Response (full):
  ...
======================================
TOOL TRACE (ToolCaseTrace)
======================================
To:              TOOL
Conversation ID: 01JD6X0YVVE97B0F8C0QXP8YT6
Auth context:    None
Created at:      2024-11-21 08:22:15.546000+00:00
Additional info:
  {}
Request:
  {"tool_name":"get_movie_suggestion", "tool_use_id":"tooluse_hahjynWhQzi86Zpwzt4Ygw", "tool_input":{"genre":"comedy"}}
Response:
  {"tool_response":{"tool_response":"The alleyways of Amsterdam (1996)"}, "latency_ms":0}
======================================
LLM TRACE (LlmCaseTrace)
======================================
To:              LLM
Conversation ID: 01JD6X0YVVE97B0F8C0QXP8YT6
Auth context:    None
Created at:      2024-11-21 08:22:17.206000+00:00
Additional info:
  {}
Request messages:
  {'toolResult': {'toolUseId': 'tooluse_hahjynWhQzi86Zpwzt4Ygw', 'status': 'success', 'content': [{'json': {'tool_response': 'The alleyways of Amsterdam (1996)'}}]}}
Response message:
  [{'text': 'For a comedy movie suggestion, how about "The Alleyways of Amsterdam" from 1996?'}]
Request (full):
  ...
Response (full):
  ...
```

What you can see is that the agent asked the user a question because it needed more information (the genre, see first trace), and the user input producer provided an answer on behalf of the user: science fiction (see second trace).

Note that you can still provide `user_inputs` in the case as well: these will be played out first, and once these are exhausted the `user_input_producer` will be invoked for getting subsequent user inputs. This way, you can 'prime' a conversation.

### 2.6 Cases with dynamic expectations

Cases can also be validated by passing it one or more validator functions. A validator function must be a Python `Callable` that accepts as input the traces of the conversation. Based on these traces the validator function should return `None` or an empty string, if the test passes. If the test fails it should return one or more messages (`str` or `Sequence[str]`).

The validator function will be invoked when the traces of the case are ran through `GenerativeAIToolkit.eval()` and this will generate measurements automatically: measurements with name `ValidationPassed` if the test passed (i.e. it returned `None` or `""`) and `ValidationFailed` otherwise. If the validation failed, the message that was returned will be included in the measurement's `additional_info` (or if an exception was thrown, the exception message):

```python
def validate_weather_report(traces: Sequence[CaseTrace]):
    last_message = traces[-1]
    if not last_message.to == "LLM":
        raise Exception("Expected last trace to be an LLM invocation") # Raising an exception works
    last_output = last_message.response["output"]["message"]["content"][0]
    if "text" not in last_output: # Returning a message works also
        return "Expected last message to contain text"
    if last_output["text"].startswith("The weather will be"):
        # Test passed!
        return
    return f"Unexpected message: {last_output["text"]}"


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

for conversation in results:
    for measurement in conversation.measurements:
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
for conversation in results:
    for conversation_traces in conversation.traces:
        trace = conversation_traces.trace
        timestamp = int(trace.trace_id.timestamp.timestamp() * 1000)
        for measurement in conversation_traces.measurements:
            # This logs the measurement in EMF to stdout:
            logger.metric(
                measurement,
                conversation_id=conversation.conversation_id,
                trace_id=conversation_traces.trace.trace_id,
                additional_info=measurement.additional_info,
                namespace="GenerativeAIToolkit",
                common_dimensions={
                    "MyCommonDimension": "MyDimensionValue"
                },
                timestamp=timestamp,
            )
```

> Note: if you run the above in AWS Lambda, the custom metrics will now be generated, because AWS Lambda writes to Amazon CloudWatch Logs automatically. Elsewhere, you would still need to send the lines from `stdout` to Amazon CloudWatch Logs.

After that, you can view the metrics in Amazon CloudWatch metrics, and you have the full functionality of Amazon CloudWatch Metrics at your disposal to graph these metrics, create alarms (e.g. based on threshold or anomaly), put on dashboards, etc:

<img src="./assets/images/sample-metric.png" alt="Sample Amazon Cloud Metric" width="1200" />

### 2.9 Deploying and Invoking the BedrockConverseAgent

The [Cookiecutter template](#cookiecutter-template) includes an AWS CDK Stack that shows how to deploy this library on AWS Lambda (as per the diagram at the top of this README):

- An AWS Lambda Function that is exposed as Function URL, so that you can use HTTP to send user input to the agent, and get a streaming response back. This Function URL has IAM auth enabled, and must be invoked with valid AWS credentials, see below. The Function URL accepts POST requests with the user input passed as body: `{"user_input": "What is the capital of France?"}`. If a conversation is to be continued, pass its ID in HTTP header `x-conversation-id`. Correspondingly when a new conversation is started, its ID will be passed back in the `x-conversation-id` response header.
- An Amazon DynamoDB table to store conversation history and traces.
- An AWS Lambda Function, that is attached to the DynamoDB table stream, to run `GenerativeAIToolkit.eval()` on the collected traces.

See [streaming-agent.ts](cookiecutter-template/{{ cookiecutter.package_name }}/lib/streaming-agent.ts).

#### Invoking the AWS Lambda Function URL with the `IamAuthInvoker`

Invoking an AWS Lambda Function URL with IAM auth entails that you must [sign the request with AWS Signature V4 as explained here](https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html).

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

### 2.10 Web UI for Conversation Debugging

The Generative AI Toolkit provides a local, web-based user interface (UI) to help you inspect and debug conversations, view evaluation results, and analyze agent behavior. This UI is particularly useful during development and testing phases, allowing you to quickly identify issues, review traces, and understand how your agent processes user queries and responds.

**Key Features:**

- **Trace Inspection:** View the entire sequence of interactions, including user messages, agent responses, and tool invocations. Traces are displayed in chronological order, accompanied by detailed metadata (timestamps, token counts, latencies, costs), making it easier to pinpoint performance bottlenecks or unexpected behaviors.
  
- **Conversation Overview:** Each conversation is presented as a cohesive flow. You can navigate through every turn in a conversation to see how the context evolves over time, how the agent utilizes tools, and how different system prompts or model parameters influence the responses.

- **Metrics and Evaluation Results:** When you run `GenerativeAIToolkit.eval()` on the collected traces, the UI provides a clear visualization of the results. This includes SQL query accuracy metrics, cost estimates, latency measurements, and custom validation checks. The UI helps you identify which cases passed or failed, and the reasons why.

- **Filtering and Sorting:** For large sets of conversations or test cases, you can filter and sort them. For example, focus on failed cases only, or examine conversations related to a specific model configuration.

Below are two example screenshots of the UI in action:

<img src="./assets/images/generative-ai-toolkit-webui-dashboard.png" alt="UI Overview Screenshot" title="UI Overview Screenshot" width="1200"/>

*In this screenshot, you can see multiple conversations along with their metrics and pass/fail status. Clicking on a conversation reveals its detailed traces and metrics.*

<img src="./assets/images/generative-ai-toolkit-webui-detail.png" alt="UI Overview Screenshot" title="UI Overview Screenshot" width="1200"/>

*Here, a single conversation’s full trace is displayed. You can see user queries, agent responses, any tool calls made, and evaluation details like latency and cost. This view helps you understand how and why the agent produced its final answer.*

**How to Launch the UI:**

After generating and evaluating traces, start the UI by calling:

```python
results.start_ui()
```

This command runs a local web server (often at http://localhost:8000) where you can interact with the web UI. When you have finished inspecting your conversations and metrics, you can shut down the UI by running:

```python
results.stop_ui()
```


The Web UI complements the command-line and code-based workflows, providing a more visual and interactive approach to debugging. By using this interface, you can refine your LLM-based application more efficiently before deploying it to production.
