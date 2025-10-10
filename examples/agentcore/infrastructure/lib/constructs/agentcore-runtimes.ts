import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { DockerImageAsset } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";

export interface AgentCoreRuntimesProps {
  agentRepository: ecr.IRepository;
  mcpServerRepository: ecr.IRepository;
  agentImageAsset: DockerImageAsset;
  mcpServerImageAsset: DockerImageAsset;
  agentExecutionRole: iam.Role;
  mcpServerExecutionRole: iam.Role;
}

export class AgentCoreRuntimes extends Construct {
  public readonly agentRuntime: bedrockagentcore.CfnRuntime;
  public readonly mcpServerRuntime: bedrockagentcore.CfnRuntime;
  public readonly agentRuntimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;
  public readonly mcpServerRuntimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;

  constructor(scope: Construct, id: string, props: AgentCoreRuntimesProps) {
    super(scope, id);

    // MCP Server Runtime (MCP protocol on port 8000)
    this.mcpServerRuntime = new bedrockagentcore.CfnRuntime(
      this,
      "McpServerRuntime",
      {
        agentRuntimeName: "agentcore_integration_mcp_server",
        description: "AgentCore runtime for the MCP server (MCP protocol)",
        roleArn: props.mcpServerExecutionRole.roleArn,
        agentRuntimeArtifact: {
          containerConfiguration: {
            containerUri: props.mcpServerImageAsset.imageUri,
          },
        },
        networkConfiguration: {
          networkMode: "PUBLIC",
        },
        environmentVariables: {
          AWS_DEFAULT_REGION: cdk.Stack.of(this).region,
        },
      }
    );

    // MCP Server Runtime Endpoint
    this.mcpServerRuntimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "McpServerRuntimeEndpoint",
      {
        name: "agentcore_integration_mcp_server_endpoint",
        description: "Runtime endpoint for the MCP server",
        agentRuntimeId: this.mcpServerRuntime.attrAgentRuntimeId,
      }
    );

    // // Agent Runtime (HTTP protocol on port 8080)
    // this.agentRuntime = new bedrockagentcore.CfnRuntime(this, "AgentRuntime", {
    //   agentRuntimeName: "agentcore_integration_agent",
    //   description: "AgentCore runtime for the agent (HTTP protocol)",
    //   roleArn: props.agentExecutionRole.roleArn,
    //   agentRuntimeArtifact: {
    //     containerConfiguration: {
    //       containerUri: props.agentImageAsset.imageUri,
    //     },
    //   },
    //   networkConfiguration: {
    //     networkMode: "PUBLIC",
    //   },
    //   environmentVariables: {
    //     AWS_DEFAULT_REGION: cdk.Stack.of(this).region,
    //     MCP_SERVER_ENDPOINT:
    //       this.mcpServerRuntimeEndpoint.attrAgentRuntimeEndpointArn,
    //   },
    // });

    // // Agent Runtime Endpoint (for direct agent invocation)
    // this.agentRuntimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
    //   this,
    //   "AgentRuntimeEndpoint",
    //   {
    //     name: "agentcore_integration_agent_endpoint",
    //     description: "Runtime endpoint for the agent",
    //     agentRuntimeId: this.agentRuntime.attrAgentRuntimeId,
    //   }
    // );

    // // Output runtime ARNs and IDs
    // new cdk.CfnOutput(this, "AgentRuntimeArn", {
    //   value: this.agentRuntime.attrAgentRuntimeArn,
    //   description: "ARN of the Agent runtime",
    // });

    // new cdk.CfnOutput(this, "AgentRuntimeId", {
    //   value: this.agentRuntime.attrAgentRuntimeId,
    //   description: "ID of the Agent runtime",
    // });

    // new cdk.CfnOutput(this, "AgentRuntimeEndpointArn", {
    //   value: this.agentRuntimeEndpoint.attrAgentRuntimeEndpointArn,
    //   description: "ARN of the Agent runtime endpoint",
    // });

    new cdk.CfnOutput(this, "McpServerRuntimeArn", {
      value: this.mcpServerRuntime.attrAgentRuntimeArn,
      description: "ARN of the MCP Server runtime",
    });

    new cdk.CfnOutput(this, "McpServerRuntimeId", {
      value: this.mcpServerRuntime.attrAgentRuntimeId,
      description: "ID of the MCP Server runtime",
    });

    new cdk.CfnOutput(this, "McpServerRuntimeEndpointArn", {
      value: this.mcpServerRuntimeEndpoint.attrAgentRuntimeEndpointArn,
      description: "ARN of the MCP Server runtime endpoint",
    });
  }
}
