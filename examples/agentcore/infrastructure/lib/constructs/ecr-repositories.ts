import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";

export interface EcrRepositoriesProps {
  agentRepositoryName: string;
}

export class EcrRepositories extends Construct {
  public readonly agentRepository: ecr.IRepository;
  public readonly agentImageAsset: DockerImageAsset;

  constructor(scope: Construct, id: string, props: EcrRepositoriesProps) {
    super(scope, id);

    // Build agent container as part of CDK deployment
    this.agentImageAsset = new DockerImageAsset(this, "AgentImageAsset", {
      directory: path.join(__dirname, "../../../agent"),
      displayName: "AgentCore Integration Agent",
      assetName: "agentcore-integration-agent",
      platform: Platform.LINUX_ARM64,
    });

    this.agentRepository = this.agentImageAsset.repository;
    new cdk.CfnOutput(this, "AgentImageUri", {
      value: this.agentImageAsset.imageUri,
      description: "URI of the built agent container image",
    });
  }
}
