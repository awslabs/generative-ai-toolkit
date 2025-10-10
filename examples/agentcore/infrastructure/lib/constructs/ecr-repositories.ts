import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";

export interface EcrRepositoriesProps {
  agentRepositoryName: string;
  mcpServerRepositoryName: string;
}

export class EcrRepositories extends Construct {
  public readonly agentRepository: ecr.IRepository;
  public readonly mcpServerRepository: ecr.IRepository;
  public readonly agentImageAsset: DockerImageAsset;
  public readonly mcpServerImageAsset: DockerImageAsset;

  constructor(scope: Construct, id: string, props: EcrRepositoriesProps) {
    super(scope, id);

    // // Build agent container as part of CDK deployment
    // // Must specify ARM64 platform for AgentCore Runtime compatibility
    // this.agentImageAsset = new DockerImageAsset(this, "AgentImageAsset", {
    //   directory: path.join(__dirname, "../../../agent"),
    //   displayName: "AgentCore Integration Agent",
    //   assetName: "agentcore-integration-agent",
    //   platform: Platform.LINUX_ARM64,
    // });

    // // Output image URIs and repository information
    // this.agentRepository = this.agentImageAsset.repository;
    // new cdk.CfnOutput(this, "AgentImageUri", {
    //   value: this.agentImageAsset.imageUri,
    //   description: "URI of the built agent container image",
    // });
    // new cdk.CfnOutput(this, "AgentRepositoryUri", {
    //   value: this.agentRepository.repositoryUri,
    //   description: "URI of the agent ECR repository",
    // });

    // Build MCP server container as part of CDK deployment
    // Must specify ARM64 platform for AgentCore Runtime compatibility
    // Using minimal Dockerfile for initial testing
    this.mcpServerImageAsset = new DockerImageAsset(
      this,
      "McpServerImageAsset",
      {
        directory: path.join(__dirname, "../../../mcp-server"),
        file: "Dockerfile.minimal",
        displayName: "AgentCore Integration MCP Server (Minimal)",
        assetName: "agentcore-integration-mcp-server",
        platform: Platform.LINUX_ARM64,
      }
    );

    // Output image URIs and repository information
    this.mcpServerRepository = this.mcpServerImageAsset.repository;
    new cdk.CfnOutput(this, "McpServerImageUri", {
      value: this.mcpServerImageAsset.imageUri,
      description: "URI of the built MCP server container image",
    });

    new cdk.CfnOutput(this, "McpServerRepositoryUri", {
      value: this.mcpServerRepository.repositoryUri,
      description: "URI of the MCP server ECR repository",
    });
  }
}
