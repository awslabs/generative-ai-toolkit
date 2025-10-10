import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export interface IamRolesProps {
  agentRepository: ecr.IRepository;
  mcpServerRepository: ecr.IRepository;
}

export class IamRoles extends Construct {
  public readonly agentExecutionRole: iam.Role;
  public readonly mcpServerExecutionRole: iam.Role;

  constructor(scope: Construct, id: string, props: IamRolesProps) {
    super(scope, id);

    // IAM execution role for Agent runtime
    this.agentExecutionRole = new iam.Role(this, "AgentExecutionRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description: "Execution role for Agent runtime",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy"
        ),
      ],
      inlinePolicies: {
        AgentPermissions: new iam.PolicyDocument({
          statements: [
            // Bedrock model access (agent needs to call LLMs)
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
              ],
              resources: [
                `arn:aws:bedrock:${
                  cdk.Stack.of(this).region
                }::foundation-model/*`,
              ],
            }),
            // CloudWatch Logs
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
              ],
              resources: [
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:/aws/agentcore/agentcore-integration-agent*`,
              ],
            }),
            // X-Ray tracing
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
              resources: ["*"],
            }),
            // // ECR access for agent container image
            // new iam.PolicyStatement({
            //   effect: iam.Effect.ALLOW,
            //   actions: [
            //     "ecr:GetAuthorizationToken",
            //     "ecr:BatchCheckLayerAvailability",
            //     "ecr:GetDownloadUrlForLayer",
            //     "ecr:BatchGetImage",
            //   ],
            //   resources: [props.agentRepository.repositoryArn],
            // }),
            // ECR authorization token (account-wide)
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["ecr:GetAuthorizationToken"],
              resources: ["*"],
            }),
            // AgentCore Runtime invocation for MCP communication
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["bedrock-agentcore:InvokeAgentRuntime"],
              resources: [
                `arn:aws:bedrock-agentcore:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:runtime-endpoint/*`,
              ],
            }),
          ],
        }),
      },
    });

    // IAM execution role for MCP Server runtime
    this.mcpServerExecutionRole = new iam.Role(this, "McpServerExecutionRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description: "Execution role for MCP Server runtime",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy"
        ),
      ],
      inlinePolicies: {
        McpServerPermissions: new iam.PolicyDocument({
          statements: [
            // CloudWatch Logs (MCP server only needs logging)
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
              ],
              resources: [
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:/aws/agentcore/agentcore-integration-mcp-server*`,
              ],
            }),
            // X-Ray tracing
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
              resources: ["*"],
            }),
            // ECR access for MCP server container image
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
              ],
              resources: [props.mcpServerRepository.repositoryArn],
            }),
            // ECR authorization token (account-wide)
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["ecr:GetAuthorizationToken"],
              resources: ["*"],
            }),
          ],
        }),
      },
    });

    // Output execution role ARNs
    new cdk.CfnOutput(this, "AgentExecutionRoleArn", {
      value: this.agentExecutionRole.roleArn,
      description: "ARN of the Agent execution role",
    });

    new cdk.CfnOutput(this, "McpServerExecutionRoleArn", {
      value: this.mcpServerExecutionRole.roleArn,
      description: "ARN of the MCP Server execution role",
    });
  }
}
