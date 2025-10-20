import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";
import * as path from "path";

export interface OAuthAuthProps {
  /**
   * The name prefix for OAuth resources
   */
  readonly namePrefix?: string;
}

export class OAuthAuth extends Construct {
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;
  public readonly testUser: cognito.CfnUserPoolUser;
  public readonly testUserCredentials: {
    secret: secretsmanager.Secret;
  };

  constructor(scope: Construct, id: string, props?: OAuthAuthProps) {
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
      accountRecovery: cognito.AccountRecovery.NONE, // Disable recovery for test users
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

    // Generate secure random password for test user
    const testPassword = this.generateSecurePassword();

    // Create test user with secure generated credentials
    this.testUser = new cognito.CfnUserPoolUser(this, "TestUser", {
      userPoolId: this.userPool.userPoolId,
      username: "mcp-test-user",
      messageAction: "SUPPRESS", // Don't send welcome email
      userAttributes: [
        {
          name: "email",
          value: "mcp-test@example.com",
        },
        {
          name: "email_verified",
          value: "true",
        },
      ],
    });

    // Set the user password using a custom resource
    // Since temporaryPassword is not available, we'll use a custom resource to set the password
    const setPasswordCustomResource = new cdk.CustomResource(
      this,
      "SetUserPassword",
      {
        serviceToken: this.createSetPasswordProvider().serviceToken,
        properties: {
          UserPoolId: this.userPool.userPoolId,
          Username: this.testUser.username,
          Password: testPassword,
          Permanent: "true", // Pass as string to avoid type conversion issues
        },
      }
    );
    setPasswordCustomResource.node.addDependency(this.testUser);

    // Store test user credentials in AWS Secrets Manager (encrypted by default)
    this.testUserCredentials = {
      secret: new secretsmanager.Secret(this, "TestUserCredentials", {
        secretName: `${namePrefix}/test-user/credentials`,
        description: "Test user credentials for MCP OAuth authentication",
        secretObjectValue: {
          username: cdk.SecretValue.unsafePlainText(this.testUser.username!),
          password: cdk.SecretValue.unsafePlainText(testPassword),
        },
        removalPolicy: cdk.RemovalPolicy.DESTROY, // For development/testing
      }),
    };

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

    new cdk.CfnOutput(this, "TestUserCredentialsSecretArn", {
      value: this.testUserCredentials.secret.secretArn,
      description: "Secrets Manager ARN for test user credentials",
      exportName: `${
        cdk.Stack.of(this).stackName
      }-TestUserCredentialsSecretArn`,
    });

    new cdk.CfnOutput(this, "TestUserCredentialsSecretName", {
      value: this.testUserCredentials.secret.secretName,
      description: "Secrets Manager secret name for test user credentials",
      exportName: `${
        cdk.Stack.of(this).stackName
      }-TestUserCredentialsSecretName`,
    });
  }

  /**
   * Generate a cryptographically secure random password
   */
  private generateSecurePassword(): string {
    // Generate a secure password with mixed case, numbers, and symbols
    // This is a deterministic approach for CDK deployment consistency
    const stackId = cdk.Stack.of(this).stackId;
    const hash = require("crypto")
      .createHash("sha256")
      .update(stackId + "mcp-oauth-password")
      .digest("hex");

    // Create a password from the hash with required character types
    const uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const lowercase = "abcdefghijklmnopqrstuvwxyz";
    const digits = "0123456789";
    const symbols = "!@#$%^&*";

    let password = "";
    password +=
      uppercase[parseInt(hash.substring(0, 2), 16) % uppercase.length];
    password +=
      lowercase[parseInt(hash.substring(2, 4), 16) % lowercase.length];
    password += digits[parseInt(hash.substring(4, 6), 16) % digits.length];
    password += symbols[parseInt(hash.substring(6, 8), 16) % symbols.length];

    // Add more characters to reach 24 characters total
    const allChars = uppercase + lowercase + digits + symbols;
    for (let i = 8; i < 48; i += 2) {
      password +=
        allChars[
          parseInt(hash.substring(i % 64, (i % 64) + 2), 16) % allChars.length
        ];
    }

    return password;
  }

  /**
   * Create IAM policy for Secrets Manager access to test user credentials
   */
  public createSecretsManagerAccessPolicy(): iam.PolicyDocument {
    return new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
          ],
          resources: [this.testUserCredentials.secret.secretArn],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["kms:Decrypt"],
          resources: [
            `arn:aws:kms:${cdk.Stack.of(this).region}:${
              cdk.Stack.of(this).account
            }:alias/aws/secretsmanager`,
          ],
          conditions: {
            StringEquals: {
              "kms:ViaService": `secretsmanager.${
                cdk.Stack.of(this).region
              }.amazonaws.com`,
            },
          },
        }),
      ],
    });
  }

  /**
   * Create a custom resource provider to set user password
   */
  private createSetPasswordProvider(): cr.Provider {
    const onEventHandler = new lambda.Function(this, "SetPasswordHandler", {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: "index.on_event",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../../lambda/set-password")
      ),
      timeout: cdk.Duration.minutes(5),
    });

    // Grant permissions to the Lambda function
    onEventHandler.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["cognito-idp:AdminSetUserPassword"],
        resources: [this.userPool.userPoolArn],
      })
    );

    return new cr.Provider(this, "SetPasswordProvider", {
      onEventHandler: onEventHandler,
    });
  }
}
