import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";
import { CognitoAuth } from "./cognito-auth";
import { RequestHeaderConfig } from "./request-header-config";

export interface AgentProps {
  /**
   * The name prefix for Agent resources
   */
  readonly namePrefix?: string;
  /**
   * The MCP Server runtime ARN for tool integration
   */
  readonly mcpServerRuntimeArn: string;
  /**
   * Cognito authentication construct for OAuth infrastructure
   */
  readonly cognitoAuth: CognitoAuth;
  /**
   * Whether to enable JWT bearer token authentication for the agent runtime
   * When enabled, the agent can be invoked with OAuth JWT tokens from Cognito
   * @default true
   */
  readonly enableJwtAuth?: boolean;
  /**
   * The Bedrock model ID to use for the agent
   * Must be available in the deployment region
   * @example "anthropic.claude-3-5-sonnet-20241022-v2:0"
   * @example "eu.anthropic.claude-3-5-sonnet-20241022-v2:0" (for EU regions)
   */
  readonly bedrockModelId: string;
}

export class Agent extends Construct {
  public readonly runtime: bedrockagentcore.CfnRuntime;
  public readonly runtimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;
  public readonly executionRole: iam.Role;
  public readonly imageAsset: DockerImageAsset;

  constructor(scope: Construct, id: string, props: AgentProps) {
    super(scope, id);

    const namePrefix = props.namePrefix || cdk.Stack.of(this).stackName;

    // Build agent container
    this.imageAsset = new DockerImageAsset(this, "ImageAsset", {
      directory: path.join(__dirname, "../../../agent"),
      displayName: "AgentCore Agent",
      assetName: "agentcore-agent",
      platform: Platform.LINUX_ARM64,
    });

    const enableJwtAuth = props.enableJwtAuth ?? true;

    // IAM execution role
    this.executionRole = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description:
        "Execution role for Agent runtime with JWT passthrough authentication",
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
                }:log-group:/aws/bedrock-agentcore/*`,
                `arn:aws:logs:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:log-group:/aws/bedrock-agentcore/*:log-stream:*`,
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
                }:trace/*`,
                `arn:aws:xray:${cdk.Stack.of(this).region}:${
                  cdk.Stack.of(this).account
                }:sampling-rule/*`,
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

    // Common environment variables for the agent runtime
    const environmentVariables = {
      AWS_DEFAULT_REGION: cdk.Stack.of(this).region,
      AWS_REGION: cdk.Stack.of(this).region,
      BEDROCK_MODEL_ID: props.bedrockModelId,
      MCP_SERVER_RUNTIME_ARN: props.mcpServerRuntimeArn,
    };

    // Build base runtime configuration
    const baseRuntimeConfig = {
      agentRuntimeName: `${namePrefix}_agent`.replace(/-/g, "_"),
      description:
        "AgentCore runtime for the agent (HTTP protocol) with JWT passthrough authentication",
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
      environmentVariables,
    };

    // Create runtime configuration with JWT authorization if enabled
    const runtimeConfig: bedrockagentcore.CfnRuntimeProps = enableJwtAuth
      ? {
          ...baseRuntimeConfig,
          authorizerConfiguration: {
            customJwtAuthorizer: {
              discoveryUrl: `https://cognito-idp.${
                cdk.Stack.of(this).region
              }.amazonaws.com/${
                props.cognitoAuth.userPool.userPoolId
              }/.well-known/openid-configuration`,
              allowedClients: [
                props.cognitoAuth.userPoolClient.userPoolClientId,
              ],
            },
          },
        }
      : baseRuntimeConfig;

    // Agent Runtime
    this.runtime = new bedrockagentcore.CfnRuntime(
      this,
      "Runtime",
      runtimeConfig
    );

    // Configure Authorization header passthrough using Custom Resource
    // This works around the CloudFormation limitation for RequestHeaderConfiguration
    new RequestHeaderConfig(this, "RequestHeaderConfig", {
      runtimeId: this.runtime.attrAgentRuntimeId,
      allowedHeaders: ["Authorization"],
      runtimeConfig: {
        containerUri: this.imageAsset.imageUri,
        roleArn: this.executionRole.roleArn,
        networkMode: "PUBLIC",
        authorizerConfiguration: enableJwtAuth
          ? {
              customJWTAuthorizer: {
                discoveryUrl: `https://cognito-idp.${
                  cdk.Stack.of(this).region
                }.amazonaws.com/${
                  props.cognitoAuth.userPool.userPoolId
                }/.well-known/openid-configuration`,
                allowedClients: [
                  props.cognitoAuth.userPoolClient.userPoolClientId,
                ],
              },
            }
          : undefined,
        environmentVariables,
        protocolConfiguration: "HTTP",
        description:
          "AgentCore runtime for the agent (HTTP protocol) with JWT passthrough authentication",
      },
    });

    // Agent Runtime Endpoint
    this.runtimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "RuntimeEndpoint",
      {
        name: `${namePrefix}_agent_endpoint`.replace(/-/g, "_"),
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
