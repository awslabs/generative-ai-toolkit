import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { EcrRepositories } from "./constructs/ecr-repositories";
import { IamRoles } from "./constructs/iam-roles";
import { CloudWatchLogs } from "./constructs/cloudwatch-logs";
import { AgentCoreRuntimes } from "./constructs/agentcore-runtimes";

export class AgentCoreIntegrationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ECR Repositories for agent container
    const ecrRepositories = new EcrRepositories(this, "EcrRepositories", {
      agentRepositoryName: "agentcore-integration-agent",
    });

    // IAM roles with principle of least privilege
    const iamRoles = new IamRoles(this, "IamRoles", {
      agentRepository: ecrRepositories.agentRepository,
    });

    // CloudWatch Log Groups
    const cloudWatchLogs = new CloudWatchLogs(this, "CloudWatchLogs", {
      agentLogGroupName: "/aws/agentcore/agentcore-integration-agent",
    });

    // AgentCore Runtimes and Endpoints
    const agentCoreRuntimes = new AgentCoreRuntimes(this, "AgentCoreRuntimes", {
      agentRepository: ecrRepositories.agentRepository,
      agentImageAsset: ecrRepositories.agentImageAsset,
      agentExecutionRole: iamRoles.agentExecutionRole,
    });
  }
}
