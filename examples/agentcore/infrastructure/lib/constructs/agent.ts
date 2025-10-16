import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";

export class Agent extends Construct {
  public readonly runtime: bedrockagentcore.CfnRuntime;
  public readonly runtimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;
  public readonly executionRole: iam.Role;
  public readonly imageAsset: DockerImageAsset;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Build agent container
    this.imageAsset = new DockerImageAsset(this, "ImageAsset", {
      directory: path.join(__dirname, "../../../agent"),
      displayName: "AgentCore Agent",
      assetName: "agentcore-agent",
      platform: Platform.LINUX_ARM64,
    });

    // IAM execution role
    this.executionRole = new iam.Role(this, "ExecutionRole", {
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
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["bedrock-agentcore:GetWorkloadAccessTokenForJWT"],
              resources: [
                `arn:aws:bedrock-agentcore:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:*`,
              ],
            }),
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
              ],
              resources: [
                "arn:aws:bedrock:*::foundation-model/*",
                `arn:aws:bedrock:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:inference-profile/*`,
              ],
            }),
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
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["logs:DescribeLogGroups"],
              resources: [
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:*`,
              ],
            }),
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
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["ecr:GetAuthorizationToken"],
              resources: ["*"],
            }),
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
              resources: [this.imageAsset.repository.repositoryArn],
            }),
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

    // Agent Runtime
    this.runtime = new bedrockagentcore.CfnRuntime(this, "Runtime", {
      agentRuntimeName: "agentcore_agent",
      description: "AgentCore runtime for the agent (HTTP protocol)",
      roleArn: this.executionRole.roleArn,
      agentRuntimeArtifact: {
        containerConfiguration: {
          containerUri: this.imageAsset.imageUri,
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

    // Agent Runtime Endpoint
    this.runtimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "RuntimeEndpoint",
      {
        name: "agentcore_agent_endpoint",
        description: "Runtime endpoint for the agent",
        agentRuntimeId: this.runtime.attrAgentRuntimeId,
      }
    );

    // Outputs
    new cdk.CfnOutput(this, "RuntimeArn", {
      value: this.runtime.attrAgentRuntimeArn,
      description: "ARN of the Agent runtime",
    });

    new cdk.CfnOutput(this, "RuntimeEndpointArn", {
      value: this.runtimeEndpoint.attrAgentRuntimeEndpointArn,
      description: "ARN of the Agent runtime endpoint",
    });

    new cdk.CfnOutput(this, "ImageUri", {
      value: this.imageAsset.imageUri,
      description: "URI of the built agent container image",
    });
  }
}
