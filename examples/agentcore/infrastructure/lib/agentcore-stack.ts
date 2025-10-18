import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { Agent } from "./constructs/agent";
import { McpServer } from "./constructs/mcp-server";
import { OAuthAuth } from "./constructs/oauth-auth";

export class AgentCoreIntegrationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // OAuth Authentication Infrastructure
    const oauthAuth = new OAuthAuth(this, "OAuthAuth", {
      namePrefix: this.stackName,
    });

    // MCP Server with OAuth authentication
    const mcpServer = new McpServer(this, "McpServer", {
      oauthAuth: oauthAuth,
      namePrefix: this.stackName,
    });

    // Agent with MCP Server integration
    const agent = new Agent(this, "Agent", {
      namePrefix: this.stackName,
      mcpServerRuntimeArn: mcpServer.runtime.attrAgentRuntimeArn,
    });
  }
}
