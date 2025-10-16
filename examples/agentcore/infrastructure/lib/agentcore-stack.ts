import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { Agent } from "./constructs/agent";
import { McpServer } from "./constructs/mcp-server";

export class AgentCoreIntegrationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Agent
    const agent = new Agent(this, "Agent");

    // MCP Server
    const mcpServer = new McpServer(this, "McpServer");
  }
}
