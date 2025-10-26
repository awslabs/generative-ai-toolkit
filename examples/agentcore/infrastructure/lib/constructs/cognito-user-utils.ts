import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";
import * as path from "path";

/**
 * Base properties for Cognito user constructs
 */
export interface BaseCognitoUserProps {
  /**
   * The Cognito User Pool where the user will be created
   */
  readonly userPool: cognito.UserPool;

  /**
   * The name prefix for user resources
   */
  readonly namePrefix?: string;

  /**
   * The username for the user
   */
  readonly username?: string;

  /**
   * The email address for the user
   */
  readonly email?: string;
}

/**
 * Configuration for creating a Cognito user
 */
export interface CognitoUserConfig {
  readonly userPool: cognito.UserPool;
  readonly username: string;
  readonly email: string;
  readonly password: string;
}

/**
 * Configuration for creating user credentials secret
 */
export interface UserCredentialsConfig {
  readonly namePrefix: string;
  readonly secretName: string;
  readonly description: string;
  readonly username: string;
  readonly password: string;
}

/**
 * Result of creating user credentials
 */
export interface UserCredentials {
  readonly secret: secretsmanager.Secret;
}

/**
 * Utility class for common Cognito user operations
 */
export class CognitoUserUtils {
  /**
   * Generate a cryptographically secure random password for a user
   */
  static generateSecurePassword(stackId: string, passwordSalt: string): string {
    // Generate a secure password with mixed case, numbers, and symbols
    // This is a deterministic approach for CDK deployment consistency
    const hash = require("crypto")
      .createHash("sha256")
      .update(stackId + passwordSalt)
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
   * Create a Cognito user with the specified configuration
   */
  static createCognitoUser(
    scope: Construct,
    id: string,
    config: CognitoUserConfig
  ): cognito.CfnUserPoolUser {
    return new cognito.CfnUserPoolUser(scope, id, {
      userPoolId: config.userPool.userPoolId,
      username: config.username,
      messageAction: "SUPPRESS", // Don't send welcome email
      userAttributes: [
        {
          name: "email",
          value: config.email,
        },
        {
          name: "email_verified",
          value: "true",
        },
      ],
    });
  }

  /**
   * Create a custom resource to set user password
   */
  static createSetPasswordCustomResource(
    scope: Construct,
    id: string,
    userPool: cognito.UserPool,
    user: cognito.CfnUserPoolUser,
    password: string
  ): cdk.CustomResource {
    const setPasswordProvider = CognitoUserUtils.createSetPasswordProvider(
      scope,
      `${id}Provider`,
      userPool
    );

    const customResource = new cdk.CustomResource(scope, id, {
      serviceToken: setPasswordProvider.serviceToken,
      properties: {
        UserPoolId: userPool.userPoolId,
        Username: user.username,
        Password: password,
        Permanent: "true", // Pass as string to avoid type conversion issues
      },
    });

    customResource.node.addDependency(user);
    return customResource;
  }

  /**
   * Save user credentials in AWS Secrets Manager
   */
  static createUserCredentials(
    scope: Construct,
    id: string,
    config: UserCredentialsConfig
  ): UserCredentials {
    const secret = new secretsmanager.Secret(scope, id, {
      secretName: `${config.namePrefix}/${config.secretName}/credentials`,
      description: config.description,
      secretObjectValue: {
        username: cdk.SecretValue.unsafePlainText(config.username),
        password: cdk.SecretValue.unsafePlainText(config.password),
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For development/testing
    });

    return { secret };
  }

  /**
   * Create IAM policy for Secrets Manager access to user credentials
   */
  static createSecretsManagerAccessPolicy(
    scope: Construct,
    secretArn: string
  ): iam.PolicyDocument {
    return new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
          ],
          resources: [secretArn],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["kms:Decrypt"],
          resources: [
            `arn:aws:kms:${cdk.Stack.of(scope).region}:${
              cdk.Stack.of(scope).account
            }:alias/aws/secretsmanager`,
          ],
          conditions: {
            StringEquals: {
              "kms:ViaService": `secretsmanager.${
                cdk.Stack.of(scope).region
              }.amazonaws.com`,
            },
          },
        }),
      ],
    });
  }

  /**
   * Create CDK outputs for user credentials
   */
  static createUserCredentialsOutputs(
    scope: Construct,
    outputPrefix: string,
    credentials: UserCredentials,
    username: string,
    description: string
  ): void {
    const stackName = cdk.Stack.of(scope).stackName;

    new cdk.CfnOutput(scope, `${outputPrefix}CredentialsSecretArn`, {
      value: credentials.secret.secretArn,
      description: `Secrets Manager ARN for ${description}`,
      exportName: `${stackName}-${outputPrefix}CredentialsSecretArn`,
    });

    new cdk.CfnOutput(scope, `${outputPrefix}CredentialsSecretName`, {
      value: credentials.secret.secretName,
      description: `Secrets Manager secret name for ${description}`,
      exportName: `${stackName}-${outputPrefix}CredentialsSecretName`,
    });

    new cdk.CfnOutput(scope, `${outputPrefix}Username`, {
      value: username,
      description: `Username for ${description}`,
      exportName: `${stackName}-${outputPrefix}Username`,
    });
  }

  /**
   * Create a custom resource provider to set user password
   */
  private static createSetPasswordProvider(
    scope: Construct,
    id: string,
    userPool: cognito.UserPool
  ): cr.Provider {
    const onEventHandler = new lambda.Function(scope, `${id}Handler`, {
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
        resources: [userPool.userPoolArn],
      })
    );

    return new cr.Provider(scope, id, {
      onEventHandler: onEventHandler,
    });
  }
}

/**
 * Constants for password generation salts
 */
export const PasswordSalts = {
  AGENT_USER: "mcp-agent-oauth-password",
  CLIENT_USER: "client-user-oauth-password",
} as const;
