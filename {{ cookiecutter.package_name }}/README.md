# {{ cookiecutter.project_name }}

This is a CDK project to build and run the Generative AI Toolkit agent "{{ cookiecutter.project_name }}".

Your next steps are to:

1. Experiment with the sample notebook to get to know Generative AI Toolkit: [`genai_toolkit_getting_started.ipynb`](`./genai_toolkit_getting_started.ipynb`)
2. Develop your own Generative AI Toolkit agent (refine system prompt, add tools, etc.): [`lib/agent/agent.py`](./lib/agent/agent.py)
3. Run sample metrics against your agent. Execute: `python evaluate_agent.py`
4. Try your agent with different models (e.g. Sonnet, Haiku): Execute: `python evaluate_agent_permutations.py`
5. Create metrics for continuous evaluation of your agent at runtime: [`lib/evaluation/measure.py`](./lib/evaluation/measure.py)
6. Deploy your agent: `cdk deploy`. This works for just the vanilla agent code as well, without needing changes from you.
7. Test the deployed agent:
  - `./test_function_url.py <lambda-function-url> 'What is the capital of France?'` - this prints a conversation ID as well as the agent's response
  - `./test_function_url.py <lambda-function-url> <conversation-id> 'What are some highlights there?'` - continue conversation

## CDK commands

* `npx cdk synth`   emits the synthesized CloudFormation template
* `npx cdk deploy`  deploy this stack to your default AWS account/region
* `npx cdk diff`    compare deployed stack with current state
* `npm run test`    perform the jest unit tests
