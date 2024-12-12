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
import { StreamingAgent } from "./streaming-agent";
import * as path from "path";

export class {{ cookiecutter.agent_name }}Stack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const AGENT_MODEL = cdk.aws_bedrock.FoundationModel.fromFoundationModelId(
      this,
      "ModelRef",
      new cdk.aws_bedrock.FoundationModelIdentifier(
        "anthropic.claude-3-sonnet-20240229-v1:0"
      )
    );

    const EVAL_MODEL = cdk.aws_bedrock.FoundationModel.fromFoundationModelId(
      this,
      "ModelRef",
      cdk.aws_bedrock.FoundationModelIdentifier
        .ANTHROPIC_CLAUDE_3_SONNET_20240229_V1_0
    );

    const agent = new StreamingAgent(this, "{{ cookiecutter.agent_name }}", {
      agentCode: cdk.aws_lambda.DockerImageCode.fromImageAsset(
        path.join(__dirname, "agent")
      ),
      evaluationCode: cdk.aws_lambda.DockerImageCode.fromImageAsset(
        path.join(__dirname, "evaluation")
      ),
      agentModels: [AGENT_MODEL],
      evaluationModels: [EVAL_MODEL],
    });
  }
}
