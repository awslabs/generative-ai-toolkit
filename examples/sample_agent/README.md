# Sample agent for tests

## Description
This is a sample agent built with the Generative AI toolkit including examples for structural tests of the agent's internal components and their interactions. It also contains a YAML-file for automating the tests as part of a CI/CD pipeline, e.g. with GitHub Actions.

## How to run it

If you want to run the agent you need:
- to get API keys for Ticketmaster, NewsAPI, OpenWeather and insert them into lib/agent/city_tools.py.
- Export variables for CONVERSATION_HISTORY_TABLE_NAME (e.g. export CONVERSATION_HISTORY_TABLE_NAME=my-dynamodb-history) and TRACES_TABLE_NAME (e.g. export TRACES_TABLE_NAME=my-dynamodb-traces) 
- set up AWS credentials for AWS_REGION (e.g. export AWS_REGION=us-east-1).
- Replace the ToDo in the file .github/workflows/full_test_run.yaml.
- Download the UNESCO world heritage site dataset from Kaggle and store it in folder **lib/data** as **UNESCO+World+Heritage+Sites.txt**.

You can then run the tests either via **pytest tests** or the YAML file as Github workflow.
