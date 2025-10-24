import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import { Construct } from "constructs";

export interface CognitoAuthProps {
  /**
   * The name prefix for Cognito resources
   */
  readonly namePrefix?: string;
}

/**
 * Cognito authentication infrastructure for OAuth flows.
 *
 * This construct creates the core Cognito infrastructure including:
 * - Cognito User Pool with secure password policies
 * - User Pool Client configured for OAuth flows
 *
 * Agent users should be created separately using the AgentUser construct
 * and passed to this construct's user pool.
 */
export class CognitoAuth extends Construct {
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;

  constructor(scope: Construct, id: string, props?: CognitoAuthProps) {
    super(scope, id);

    const namePrefix = props?.namePrefix || cdk.Stack.of(this).stackName;

    // Create Cognito User Pool with appropriate password policies
    this.userPool = new cognito.UserPool(this, "UserPool", {
      userPoolName: `${namePrefix}-user-pool`,
      selfSignUpEnabled: false, // Only admin-created users
      signInAliases: {
        username: true,
      },
      passwordPolicy: {
        minLength: 12,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      accountRecovery: cognito.AccountRecovery.NONE, // Disable recovery for agent users
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For development/testing
    });

    // Create Cognito App Client configured for public client OAuth flows
    this.userPoolClient = new cognito.UserPoolClient(this, "UserPoolClient", {
      userPool: this.userPool,
      userPoolClientName: `${namePrefix}-client`,
      generateSecret: false, // Public client for OAuth flows
      authFlows: {
        userPassword: true, // Enable USER_PASSWORD_AUTH flow
        userSrp: true, // Enable SRP authentication
      },
      accessTokenValidity: cdk.Duration.hours(1), // 1 hour access token
      refreshTokenValidity: cdk.Duration.days(30), // 30 days refresh token
      preventUserExistenceErrors: true, // Security best practice
    });

    // Add CDK outputs for User Pool ID, Client ID, and Discovery URL
    new cdk.CfnOutput(this, "UserPoolId", {
      value: this.userPool.userPoolId,
      description: "Cognito User Pool ID for OAuth authentication",
      exportName: `${cdk.Stack.of(this).stackName}-UserPoolId`,
    });

    new cdk.CfnOutput(this, "UserPoolClientId", {
      value: this.userPoolClient.userPoolClientId,
      description: "Cognito User Pool Client ID for OAuth authentication",
      exportName: `${cdk.Stack.of(this).stackName}-UserPoolClientId`,
    });

    new cdk.CfnOutput(this, "UserPoolDiscoveryUrl", {
      value: `https://cognito-idp.${cdk.Stack.of(this).region}.amazonaws.com/${
        this.userPool.userPoolId
      }/.well-known/openid_configuration`,
      description: "OAuth Discovery URL for the Cognito User Pool",
      exportName: `${cdk.Stack.of(this).stackName}-UserPoolDiscoveryUrl`,
    });
  }
}
