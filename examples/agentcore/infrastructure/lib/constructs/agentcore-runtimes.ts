import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { DockerImageAsset } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";

export interface AgentCoreRuntimesProps {
  agentRepository: ecr.IRepository;
  agentImageAsset: DockerImageAsset;
  agentExecutionRole: iam.Role;
}

export class AgentCoreRuntimes extends Construct {
  public readonly agentRuntime: bedrockagentcore.CfnRuntime;
  public readonly agentRuntimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;

  constructor(scope: Construct, id: string, props: AgentCoreRuntimesProps) {
    super(scope, id);

    // Agent Runtime (HTTP protocol on port 8080)
    this.agentRuntime = new bedrockagentcore.CfnRuntime(this, "AgentRuntime", {
      agentRuntimeName: "agentcore_integration_agent",
      description: "AgentCore runtime for the agent (HTTP protocol)",
      roleArn: props.agentExecutionRole.roleArn,
      agentRuntimeArtifact: {
        containerConfiguration: {
          containerUri: props.agentImageAsset.imageUri,
        },
      },
      networkConfiguration: {
        networkMode: "PUBLIC",
      },
      protocolConfiguration: "HTTP",
      environmentVariables: {
        AWS_DEFAULT_REGION: cdk.Stack.of(this).region,
        AWS_REGION: cdk.Stack.of(this).region,
        BEDROCK_MODEL_ID: "eu.anthropic.claude-sonnet-4-20250514-v1:0",
      },
    });

    // Agent Runtime Endpoint (for direct agent invocation)
    this.agentRuntimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "AgentRuntimeEndpoint",
      {
        name: "agentcore_integration_agent_endpoint",
        description: "Runtime endpoint for the agent",
        agentRuntimeId: this.agentRuntime.attrAgentRuntimeId,
      }
    );

    // Output runtime ARNs and IDs
    new cdk.CfnOutput(this, "AgentRuntimeArn", {
      value: this.agentRuntime.attrAgentRuntimeArn,
      description: "ARN of the Agent runtime",
    });

    new cdk.CfnOutput(this, "AgentRuntimeId", {
      value: this.agentRuntime.attrAgentRuntimeId,
      description: "ID of the Agent runtime",
    });

    new cdk.CfnOutput(this, "AgentRuntimeEndpointArn", {
      value: this.agentRuntimeEndpoint.attrAgentRuntimeEndpointArn,
      description: "ARN of the Agent runtime endpoint",
    });
  }
}
