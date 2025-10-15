import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export interface IamRolesProps {
  agentRepository: ecr.IRepository;
}

export class IamRoles extends Construct {
  public readonly agentExecutionRole: iam.Role;

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
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "CloudWatchLambdaApplicationSignalsExecutionRolePolicy"
        ),
      ],
      inlinePolicies: {
        AgentPermissions: new iam.PolicyDocument({
          statements: [
            // AgentCore workload access token
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["bedrock-agentcore:GetWorkloadAccessTokenForJWT"],
              resources: [
                `arn:aws:bedrock-agentcore:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:*`,
              ],
            }),
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
                "logs:DescribeLogStreams",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
              ],
              resources: [
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:/aws/bedrock-agentcore/runtimes/*`,
              ],
            }),
            // CloudWatch Logs - describe log groups
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["logs:DescribeLogGroups"],
              resources: [
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:*`,
              ],
            }),
            // X-Ray tracing
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords",
                "xray:GetSamplingRules",
                "xray:GetSamplingTargets",
              ],
              resources: [
                `arn:aws:xray:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:*`,
              ],
            }),
            // CloudWatch Metrics
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["cloudwatch:PutMetricData"],
              resources: [
                `arn:aws:cloudwatch:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:*`,
              ],
              conditions: {
                StringEquals: {
                  "cloudwatch:namespace": "bedrock-agentcore",
                },
              },
            }),
            // ECR authorization token (account-wide)
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["ecr:GetAuthorizationToken"],
              resources: ["*"],
            }),
            // ECR image access
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
              ],
              resources: [props.agentRepository.repositoryArn],
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

    // Output execution role ARN
    new cdk.CfnOutput(this, "AgentExecutionRoleArn", {
      value: this.agentExecutionRole.roleArn,
      description: "ARN of the Agent execution role",
    });
  }
}
