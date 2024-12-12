/**
 * Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { NagSuppressions } from "cdk-nag";

export interface StreamingAgentProps {
  agentCode: cdk.aws_lambda.DockerImageCode;
  agentModels: cdk.aws_bedrock.FoundationModel[];
  evaluationCode: cdk.aws_lambda.DockerImageCode;
  evaluationModels?: cdk.aws_bedrock.FoundationModel[];
}

export class StreamingAgent extends Construct {
  agentFn: cdk.aws_lambda.Function;
  evaluationFn: cdk.aws_lambda.Function;
  table: cdk.aws_dynamodb.Table;
  lambdaUrl: cdk.aws_lambda.FunctionUrl;
  samplingRateParameter: cdk.aws_ssm.StringParameter;
  constructor(scope: Construct, id: string, props: StreamingAgentProps) {
    super(scope, id);

    // Table for conversation history and traces
    const agentTable = new cdk.aws_dynamodb.Table(this, "Table", {
      partitionKey: {
        name: "pk",
        type: cdk.aws_dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: "sk",
        type: cdk.aws_dynamodb.AttributeType.STRING,
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      stream: cdk.aws_dynamodb.StreamViewType.NEW_IMAGE,
      pointInTimeRecovery: true,
    });
    this.table = agentTable;
    agentTable.addGlobalSecondaryIndex({
      indexName: "traces",
      partitionKey: {
        name: "conversation_id",
        type: cdk.aws_dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: "trace_id",
        type: cdk.aws_dynamodb.AttributeType.STRING,
      },
      projectionType: cdk.aws_dynamodb.ProjectionType.KEYS_ONLY,
    });
    agentTable.addGlobalSecondaryIndex({
      indexName: "messages",
      partitionKey: {
        name: "conversation_id",
        type: cdk.aws_dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: "msg_nr",
        type: cdk.aws_dynamodb.AttributeType.NUMBER,
      },
      projectionType: cdk.aws_dynamodb.ProjectionType.KEYS_ONLY,
    });
    // Lambda function that implements the agent
    const fn = new cdk.aws_lambda.DockerImageFunction(this, "Agent", {
      code: props.agentCode,
      architecture: cdk.aws_lambda.Architecture.X86_64,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(5),
      environment: {
        AWS_LWA_INVOKE_MODE: "RESPONSE_STREAM",
        CONVERSATION_HISTORY_TABLE_NAME: agentTable.tableName,
        TRACES_TABLE_NAME: agentTable.tableName,
      },
    });
    this.agentFn = fn;
    agentTable.grantReadWriteData(fn);
    NagSuppressions.addResourceSuppressions(
      fn,
      [
        {
          id: "AwsSolutions-IAM5",
          reason: "Allow index usage, which needs /index/*",
          appliesTo: [
            `Resource::<${cdk.Stack.of(this).getLogicalId(
              this.table.node.defaultChild as cdk.aws_dynamodb.CfnTable
            )}.Arn>/index/*`,
          ],
        },
      ],
      true
    );

    fn.addToRolePolicy(
      new cdk.aws_iam.PolicyStatement({
        actions: [
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:InvokeModel",
        ],
        resources: props.agentModels.map((m) => m.modelArn),
        effect: cdk.aws_iam.Effect.ALLOW,
      })
    );

    const lambdaUrl = fn.addFunctionUrl({
      authType: cdk.aws_lambda.FunctionUrlAuthType.AWS_IAM,
      invokeMode: cdk.aws_lambda.InvokeMode.RESPONSE_STREAM,
    });
    this.lambdaUrl = lambdaUrl;
    new cdk.CfnOutput(this, "FunctionUrl ", { value: lambdaUrl.url });

    // SSM param for sampling rate (to be expressed as a percentage)
    const samplingRateParam = new cdk.aws_ssm.StringParameter(
      this,
      "GenAIToolkitEvaluationSamplingRate",
      {
        stringValue: "100",
        allowedPattern: "^\\d+$",
      }
    );
    this.samplingRateParameter = samplingRateParam;

    // Lambda function to run GenerativeAIToolkit eval in traces in DynamoDB
    const evalFn = new cdk.aws_lambda.DockerImageFunction(this, "TraceEval", {
      code: props.evaluationCode,
      architecture: cdk.aws_lambda.Architecture.X86_64,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(2),
      environment: {
        TRACES_TABLE_NAME: agentTable.tableName,
        SAMPLING_RATE_PARAM_NAME: samplingRateParam.parameterName,
      },
    });
    this.evaluationFn = evalFn;
    samplingRateParam.grantRead(evalFn);
    new cdk.CfnOutput(this, "SamplingRateParameter ", {
      value: samplingRateParam.parameterName,
    });

    if (props.evaluationModels) {
      evalFn.addToRolePolicy(
        new cdk.aws_iam.PolicyStatement({
          actions: [
            "bedrock:InvokeModelWithResponseStream",
            "bedrock:InvokeModel",
          ],
          resources: props.evaluationModels.map((m) => m.modelArn),
          effect: cdk.aws_iam.Effect.ALLOW,
        })
      );
    }

    evalFn.addEventSource(
      new cdk.aws_lambda_event_sources.DynamoEventSource(agentTable, {
        startingPosition: cdk.aws_lambda.StartingPosition.LATEST,
        batchSize: 100,
        maxBatchingWindow: cdk.Duration.seconds(60),
      })
    );
    NagSuppressions.addResourceSuppressions(
      this.evaluationFn,
      [
        {
          id: "AwsSolutions-IAM5",
          reason:
            "Need to be able to do dynamodb:ListStreams on * to make the stream event source work",
          appliesTo: ["Resource::*"],
        },
      ],
      true
    );

    NagSuppressions.addResourceSuppressions(
      this,
      [
        {
          id: "AwsSolutions-IAM4",
          reason: "Allow use of AWSLambdaBasicExecutionRole",
          appliesTo: [
            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
          ],
        },
      ],
      true
    );
  }
}
