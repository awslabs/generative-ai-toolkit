import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { EcrRepositories } from "./constructs/ecr-repositories";
import { IamRoles } from "./constructs/iam-roles";
import { CloudWatchLogs } from "./constructs/cloudwatch-logs";
import { AgentCoreRuntimes } from "./constructs/agentcore-runtimes";

export class AgentCoreIntegrationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ECR Repositories for agent and MCP server containers
    const ecrRepositories = new EcrRepositories(this, "EcrRepositories", {
      agentRepositoryName: "agentcore-integration-agent",
      mcpServerRepositoryName: "agentcore-integration-mcp-server",
    });

    // IAM roles with principle of least privilege
    const iamRoles = new IamRoles(this, "IamRoles", {
      agentRepository: ecrRepositories.agentRepository,
      mcpServerRepository: ecrRepositories.mcpServerRepository,
    });

    // CloudWatch Log Groups
    const cloudWatchLogs = new CloudWatchLogs(this, "CloudWatchLogs", {
      agentLogGroupName: "/aws/agentcore/agentcore-integration-agent",
      mcpServerLogGroupName: "/aws/agentcore/agentcore-integration-mcp-server",
    });

    // AgentCore Runtimes and Endpoints
    const agentCoreRuntimes = new AgentCoreRuntimes(this, "AgentCoreRuntimes", {
      agentRepository: ecrRepositories.agentRepository,
      mcpServerRepository: ecrRepositories.mcpServerRepository,
      agentImageAsset: ecrRepositories.agentImageAsset,
      mcpServerImageAsset: ecrRepositories.mcpServerImageAsset,
      agentExecutionRole: iamRoles.agentExecutionRole,
      mcpServerExecutionRole: iamRoles.mcpServerExecutionRole,
    });
  }
}
