import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as custom from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export interface RequestHeaderConfigProps {
  /**
   * The AgentCore Runtime ID to configure
   */
  runtimeId: string;

  /**
   * List of headers to allow through to the runtime
   */
  allowedHeaders: string[];

  /**
   * The current runtime configuration (used to detect changes)
   */
  runtimeConfig: {
    containerUri: string;
    roleArn: string;
    networkMode: string;
    authorizerConfiguration?: any;
    environmentVariables?: { [key: string]: string };
    protocolConfiguration?: string;
    description?: string;
  };
}

/**
 * Custom Resource to configure RequestHeaderAllowlist for AgentCore Runtime
 *
 * This works around the limitation that CloudFormation doesn't yet support
 * the RequestHeaderConfiguration property for AWS::BedrockAgentCore::Runtime
 *
 * Uses AwsCustomResource for simplified implementation without Lambda functions
 */
export class RequestHeaderConfig extends Construct {
  constructor(scope: Construct, id: string, props: RequestHeaderConfigProps) {
    super(scope, id);

    // Prepare update parameters
    const updateParams = {
      agentRuntimeId: props.runtimeId,
      agentRuntimeArtifact: {
        containerConfiguration: {
          containerUri: props.runtimeConfig.containerUri,
        },
      },
      roleArn: props.runtimeConfig.roleArn,
      networkConfiguration: {
        networkMode: props.runtimeConfig.networkMode,
      },
      requestHeaderConfiguration: {
        requestHeaderAllowlist: props.allowedHeaders,
      },
      // Add optional parameters
      ...(props.runtimeConfig.authorizerConfiguration && {
        authorizerConfiguration: props.runtimeConfig.authorizerConfiguration,
      }),
      ...(props.runtimeConfig.environmentVariables && {
        environmentVariables: props.runtimeConfig.environmentVariables,
      }),
      ...(props.runtimeConfig.protocolConfiguration && {
        protocolConfiguration: {
          serverProtocol: props.runtimeConfig.protocolConfiguration,
        },
      }),
      ...(props.runtimeConfig.description && {
        description: props.runtimeConfig.description,
      }),
    };

    // Create AwsCustomResource for updating runtime configuration
    new custom.AwsCustomResource(this, "Resource", {
      onCreate: {
        service: "bedrock-agentcore-control",
        action: "UpdateAgentRuntime",
        parameters: updateParams,
        physicalResourceId: custom.PhysicalResourceId.of(
          `request-header-config-${props.runtimeId}`
        ),
      },
      onUpdate: {
        service: "bedrock-agentcore-control",
        action: "UpdateAgentRuntime",
        parameters: updateParams,
        physicalResourceId: custom.PhysicalResourceId.of(
          `request-header-config-${props.runtimeId}`
        ),
      },
      // No onDelete needed - we leave the configuration in place
      policy: custom.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock-agentcore:UpdateAgentRuntime",
            "bedrock-agentcore:GetAgentRuntime",
          ],
          resources: [
            `arn:aws:bedrock-agentcore:${cdk.Stack.of(this).region}:${
              cdk.Stack.of(this).account
            }:runtime/${props.runtimeId}`,
          ],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["iam:PassRole"],
          resources: [props.runtimeConfig.roleArn],
        }),
      ]),
      installLatestAwsSdk: true,
      timeout: cdk.Duration.minutes(5),
    });
  }
}
