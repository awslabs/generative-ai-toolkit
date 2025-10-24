import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import {
  BaseCognitoUserProps,
  CognitoUserUtils,
  PasswordSalts,
  UserCredentials,
} from "./cognito-user-utils";

export interface ClientUserProps extends BaseCognitoUserProps {
  /**
   * The username for the client user
   * @default "agent-client-user"
   */
  readonly username?: string;

  /**
   * The email address for the client user
   * @default "agent-client@example.com"
   */
  readonly email?: string;
}

/**
 * Creates a client user in Cognito User Pool for invoking the agent runtime with OAuth tokens.
 *
 * This construct creates a user account that can invoke the agent runtime "with an OAuth
 * compliant access token using JWT format" as described in the AWS documentation.
 * The credentials are securely stored in AWS Secrets Manager.
 *
 * This user is different from the AgentUser - while AgentUser is used by agent.py to
 * authenticate with mcp_server.py, this ClientUser is used by external clients to
 * invoke the agent runtime itself using JWT bearer tokens.
 */
export class ClientUser extends Construct {
  public readonly user: cognito.CfnUserPoolUser;
  public readonly credentials: UserCredentials;

  constructor(scope: Construct, id: string, props: ClientUserProps) {
    super(scope, id);

    const namePrefix = props.namePrefix || cdk.Stack.of(this).stackName;
    const username = props.username || "agent-client-user";
    const email = props.email || "agent-client@example.com";

    // Generate secure random password for client user
    const clientPassword = CognitoUserUtils.generateSecurePassword(
      cdk.Stack.of(this).stackId,
      PasswordSalts.CLIENT_USER
    );

    // Create client user with secure generated credentials
    this.user = CognitoUserUtils.createCognitoUser(this, "ClientUser", {
      userPool: props.userPool,
      username: username,
      email: email,
      password: clientPassword,
    });

    // Set the user password using a custom resource
    CognitoUserUtils.createSetPasswordCustomResource(
      this,
      "SetClientUserPassword",
      props.userPool,
      this.user,
      clientPassword
    );

    // Store client user credentials in AWS Secrets Manager (encrypted by default)
    this.credentials = CognitoUserUtils.createUserCredentials(
      this,
      "ClientUserCredentials",
      {
        namePrefix: namePrefix,
        secretName: "client-user",
        description:
          "Client user credentials for invoking agent runtime with OAuth JWT bearer tokens",
        username: username,
        password: clientPassword,
      }
    );

    // Add CDK outputs for client user credentials
    CognitoUserUtils.createUserCredentialsOutputs(
      this,
      "ClientUser",
      this.credentials,
      username,
      "client user credentials used to invoke agent runtime with JWT tokens"
    );
  }

  /**
   * Create IAM policy for Secrets Manager access to client user credentials
   */
  public createSecretsManagerAccessPolicy(): iam.PolicyDocument {
    return CognitoUserUtils.createSecretsManagerAccessPolicy(
      this,
      this.credentials.secret.secretArn
    );
  }
}
