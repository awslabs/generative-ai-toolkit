import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { Agent } from "./constructs/agent";
import { ClientUser } from "./constructs/client-user";
import { CognitoAuth } from "./constructs/cognito-auth";
import { McpServer } from "./constructs/mcp-server";

export class AgentCoreIntegrationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Claude Sonnet 4 model lookup table by region
    // Prefers Global inference profile where available for better performance and availability
    const claudeSonnet4ModelLookup: Record<string, string> = {
      // Global inference profile supported regions (preferred)
      "us-west-2": "global.anthropic.claude-sonnet-4-20250514-v1:0",
      "us-east-1": "global.anthropic.claude-sonnet-4-20250514-v1:0",
      "us-east-2": "global.anthropic.claude-sonnet-4-20250514-v1:0",
      "eu-west-1": "global.anthropic.claude-sonnet-4-20250514-v1:0",
      "ap-northeast-1": "global.anthropic.claude-sonnet-4-20250514-v1:0",

      // Regional models for other popular regions
      "eu-central-1": "eu.anthropic.claude-sonnet-4-20250514-v1:0",
      "eu-west-2": "eu.anthropic.claude-sonnet-4-20250514-v1:0",
      "ap-southeast-1": "anthropic.claude-sonnet-4-20250514-v1:0",
      "ap-southeast-2": "anthropic.claude-sonnet-4-20250514-v1:0",
      "ca-central-1": "anthropic.claude-sonnet-4-20250514-v1:0",
    };

    // Get the model ID based on the deployment region
    const deploymentRegion = cdk.Stack.of(this).region;
    const bedrockModelId = claudeSonnet4ModelLookup[deploymentRegion];

    if (!bedrockModelId) {
      throw new Error(
        `❌ Claude Sonnet 4 is not supported in region: ${deploymentRegion}\n` +
          `Supported regions: ${Object.keys(claudeSonnet4ModelLookup).join(
            ", "
          )}\n` +
          `Please deploy to one of the supported regions or update the model lookup table.`
      );
    }

    console.log(
      `Using Claude Sonnet 4 model: ${bedrockModelId} (region: ${deploymentRegion})`
    );

    // Cognito Authentication Infrastructure
    const cognitoAuth = new CognitoAuth(this, "CognitoAuth", {
      namePrefix: this.stackName,
    });

    // Client User for invoking agent runtime with JWT tokens
    // This user is used by external clients to invoke the agent runtime with OAuth JWT bearer tokens
    const clientUser = new ClientUser(this, "ClientUser", {
      userPool: cognitoAuth.userPool,
      namePrefix: this.stackName,
    });

    // MCP Server with Cognito authentication (JWT passthrough)
    const mcpServer = new McpServer(this, "McpServer", {
      cognitoAuth: cognitoAuth,
      namePrefix: this.stackName,
    });

    // Agent with MCP Server integration and JWT passthrough authentication
    const agent = new Agent(this, "Agent", {
      namePrefix: this.stackName,
      mcpServerRuntimeArn: mcpServer.runtime.attrAgentRuntimeArn,
      cognitoAuth: cognitoAuth,
      enableJwtAuth: true, // Enable JWT bearer token authentication
      bedrockModelId: bedrockModelId, // Required model ID
    });
  }
}
