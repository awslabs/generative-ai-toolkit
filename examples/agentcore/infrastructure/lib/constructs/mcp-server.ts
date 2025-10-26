import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";
import { CognitoAuth } from "./cognito-auth";

export interface McpServerProps {
  /**
   * Cognito authentication construct for MCP server authorization
   */
  readonly cognitoAuth?: CognitoAuth;
  /**
   * The name prefix for MCP Server resources
   */
  readonly namePrefix?: string;
}

export class McpServer extends Construct {
  public readonly runtime: bedrockagentcore.CfnRuntime;
  public readonly runtimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;
  public readonly executionRole: iam.Role;
  public readonly imageAsset: DockerImageAsset;
  public readonly cognitoAuth?: CognitoAuth;

  constructor(scope: Construct, id: string, props?: McpServerProps) {
    super(scope, id);

    this.cognitoAuth = props?.cognitoAuth;
    const namePrefix = props?.namePrefix || cdk.Stack.of(this).stackName;

    // Build MCP server container
    this.imageAsset = new DockerImageAsset(this, "ImageAsset", {
      directory: path.join(__dirname, "../../../mcp-server"),
      displayName: "AgentCore MCP Server",
      assetName: "agentcore-mcp-server",
      platform: Platform.LINUX_ARM64,
    });

    // Build base IAM policy statements
    // Note: Workload identity permissions (GetWorkloadAccessTokenForJWT) are automatically
    // handled by the AWSServiceRoleForBedrockAgentCoreRuntimeIdentity Service-Linked Role
    // for agents created on or after October 13, 2025
    const baseStatements = [
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
          }:log-group:/aws/bedrock-agentcore/*`,
        ],
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
    ];

    // Build inline policies
    const inlinePolicies: { [name: string]: iam.PolicyDocument } = {
      McpServerPermissions: new iam.PolicyDocument({
        statements: baseStatements,
      }),
    };

    // IAM execution role
    this.executionRole = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description:
        "Execution role for MCP Server runtime with JWT passthrough authentication",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy"
        ),
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "CloudWatchLambdaApplicationSignalsExecutionRolePolicy"
        ),
      ],
      inlinePolicies,
    });

    // Build base runtime configuration
    const baseRuntimeConfig = {
      agentRuntimeName: `${namePrefix}_mcp_server`.replace(/-/g, "_"),
      description:
        "AgentCore runtime for the MCP server (MCP protocol) with JWT passthrough authentication",
      roleArn: this.executionRole.roleArn,
      agentRuntimeArtifact: {
        containerConfiguration: {
          containerUri: this.imageAsset.imageUri,
        },
      },
      networkConfiguration: {
        networkMode: "PUBLIC",
      },
      protocolConfiguration: "MCP",
    };

    // Create runtime configuration with JWT authorization if Cognito is enabled
    const runtimeConfig: bedrockagentcore.CfnRuntimeProps = this.cognitoAuth
      ? {
          ...baseRuntimeConfig,
          authorizerConfiguration: {
            customJwtAuthorizer: {
              discoveryUrl: `https://cognito-idp.${
                cdk.Stack.of(this).region
              }.amazonaws.com/${
                this.cognitoAuth.userPool.userPoolId
              }/.well-known/openid-configuration`,
              allowedClients: [
                this.cognitoAuth.userPoolClient.userPoolClientId,
              ],
            },
          },
        }
      : baseRuntimeConfig;

    // MCP Server Runtime
    this.runtime = new bedrockagentcore.CfnRuntime(
      this,
      "Runtime",
      runtimeConfig
    );

    // MCP Server Runtime Endpoint
    this.runtimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "RuntimeEndpoint",
      {
        name: `${namePrefix}_mcp_server_endpoint`.replace(/-/g, "_"),
        description:
          "Runtime endpoint for the MCP server with JWT passthrough authentication",
        agentRuntimeId: this.runtime.attrAgentRuntimeId,
      }
    );

    // Outputs
    new cdk.CfnOutput(this, "RuntimeArn", {
      value: this.runtime.attrAgentRuntimeArn,
      description: "ARN of the MCP Server runtime",
    });

    new cdk.CfnOutput(this, "RuntimeEndpointArn", {
      value: this.runtimeEndpoint.attrAgentRuntimeEndpointArn,
      description: "ARN of the MCP Server runtime endpoint",
    });

    new cdk.CfnOutput(this, "ImageUri", {
      value: this.imageAsset.imageUri,
      description: "URI of the built MCP server container image",
    });
  }
}
