import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";
import { OAuthAuth } from "./oauth-auth";

export interface McpServerProps {
  /**
   * OAuth authentication construct for MCP server authorization
   */
  readonly oauthAuth?: OAuthAuth;
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
  public readonly oauthAuth?: OAuthAuth;

  constructor(scope: Construct, id: string, props?: McpServerProps) {
    super(scope, id);

    this.oauthAuth = props?.oauthAuth;
    const namePrefix = props?.namePrefix || cdk.Stack.of(this).stackName;

    // Build MCP server container
    this.imageAsset = new DockerImageAsset(this, "ImageAsset", {
      directory: path.join(__dirname, "../../../mcp-server"),
      displayName: "AgentCore MCP Server",
      assetName: "agentcore-mcp-server",
      platform: Platform.LINUX_ARM64,
    });

    // Build base IAM policy statements
    const baseStatements = [
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
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      }),
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
        resources: [this.imageAsset.repository.repositoryArn],
      }),
    ];

    // Add OAuth-specific permissions if OAuth is enabled
    const inlinePolicies: { [name: string]: iam.PolicyDocument } = {
      McpServerPermissions: new iam.PolicyDocument({
        statements: baseStatements,
      }),
    };

    if (this.oauthAuth) {
      inlinePolicies.SecretsManagerAccess =
        this.oauthAuth.createSecretsManagerAccessPolicy();
    }

    // IAM execution role
    this.executionRole = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description: "Execution role for MCP Server runtime",
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
        "AgentCore runtime for the MCP server (MCP protocol) with OAuth authentication",
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

    // Create runtime configuration with JWT authorization if OAuth is enabled
    const runtimeConfig: bedrockagentcore.CfnRuntimeProps = this.oauthAuth
      ? {
          ...baseRuntimeConfig,
          authorizerConfiguration: {
            customJwtAuthorizer: {
              discoveryUrl: `https://cognito-idp.${
                cdk.Stack.of(this).region
              }.amazonaws.com/${
                this.oauthAuth.userPool.userPoolId
              }/.well-known/openid-configuration`,
              allowedClients: [this.oauthAuth.userPoolClient.userPoolClientId],
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
          "Runtime endpoint for the MCP server with OAuth authentication",
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
