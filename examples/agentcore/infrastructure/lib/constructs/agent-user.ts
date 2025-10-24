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

export interface AgentUserProps extends BaseCognitoUserProps {
  /**
   * The username for the agent user
   * @default "mcp-agent-user"
   */
  readonly username?: string;

  /**
   * The email address for the agent user
   * @default "mcp-agent@example.com"
   */
  readonly email?: string;
}

/**
 * Creates an agent user in Cognito User Pool for MCP server authentication.
 *
 * This construct creates a dedicated user account that the agent (examples/agentcore/agent/agent.py)
 * uses to authenticate with the MCP server (examples/agentcore/mcp-server/mcp_server.py) via OAuth.
 * The credentials are securely stored in AWS Secrets Manager.
 *
 */
export class AgentUser extends Construct {
  public readonly user: cognito.CfnUserPoolUser;
  public readonly credentials: UserCredentials;

  constructor(scope: Construct, id: string, props: AgentUserProps) {
    super(scope, id);

    const namePrefix = props.namePrefix || cdk.Stack.of(this).stackName;
    const username = props.username || "mcp-agent-user";
    const email = props.email || "mcp-agent@example.com";

    // Generate secure random password for agent user
    const agentPassword = CognitoUserUtils.generateSecurePassword(
      cdk.Stack.of(this).stackId,
      PasswordSalts.AGENT_USER
    );

    // Create agent user with secure generated credentials
    this.user = CognitoUserUtils.createCognitoUser(this, "AgentUser", {
      userPool: props.userPool,
      username: username,
      email: email,
      password: agentPassword,
    });

    // Set the user password using a custom resource
    CognitoUserUtils.createSetPasswordCustomResource(
      this,
      "SetAgentUserPassword",
      props.userPool,
      this.user,
      agentPassword
    );

    // Store agent user credentials in AWS Secrets Manager (encrypted by default)
    this.credentials = CognitoUserUtils.createUserCredentials(
      this,
      "AgentUserCredentials",
      {
        namePrefix: namePrefix,
        secretName: "agent-user",
        description:
          "Agent user credentials for MCP OAuth authentication between agent.py and mcp_server.py",
        username: username,
        password: agentPassword,
      }
    );

    // Add CDK outputs for agent user credentials
    CognitoUserUtils.createUserCredentialsOutputs(
      this,
      "AgentUser",
      this.credentials,
      username,
      "agent user credentials used by agent.py to access mcp_server.py"
    );
  }

  /**
   * Create IAM policy for Secrets Manager access to agent user credentials
   */
  public createSecretsManagerAccessPolicy(): iam.PolicyDocument {
    return CognitoUserUtils.createSecretsManagerAccessPolicy(
      this,
      this.credentials.secret.secretArn
    );
  }
}
