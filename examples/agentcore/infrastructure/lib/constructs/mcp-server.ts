import * as cdk from "aws-cdk-lib";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";

export class McpServer extends Construct {
  public readonly runtime: bedrockagentcore.CfnRuntime;
  public readonly runtimeEndpoint: bedrockagentcore.CfnRuntimeEndpoint;
  public readonly executionRole: iam.Role;
  public readonly imageAsset: DockerImageAsset;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Build MCP server container
    this.imageAsset = new DockerImageAsset(this, "ImageAsset", {
      directory: path.join(__dirname, "../../../mcp-server"),
      displayName: "AgentCore MCP Server",
      assetName: "agentcore-mcp-server",
      platform: Platform.LINUX_ARM64,
    });

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
      inlinePolicies: {
        McpServerPermissions: new iam.PolicyDocument({
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
          ],
        }),
      },
    });

    // MCP Server Runtime
    this.runtime = new bedrockagentcore.CfnRuntime(this, "Runtime", {
      agentRuntimeName: "agentcore_mcp_server",
      description: "AgentCore runtime for the MCP server (MCP protocol)",
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
    });

    // MCP Server Runtime Endpoint
    this.runtimeEndpoint = new bedrockagentcore.CfnRuntimeEndpoint(
      this,
      "RuntimeEndpoint",
      {
        name: "agentcore_mcp_server_endpoint",
        description: "Runtime endpoint for the MCP server",
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
