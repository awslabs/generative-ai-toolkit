#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import * as fs from "fs";
import * as path from "path";
import { AwsSolutionsChecks } from "cdk-nag";
import { AgentCoreIntegrationStack } from "../lib/agentcore-stack";
import { CdkNagSuppressions } from "../lib/cdk-nag-suppressions";

function getStackName(): string {
  // Priority order:
  // 1. Explicit environment variable
  if (process.env.CDK_STACK_NAME) {
    return process.env.CDK_STACK_NAME;
  }

  // 2. Auto-generate from username
  const username = process.env.USER || process.env.USERNAME || "dev";
  return `${username}-agentcore-stack`;
}

function writeStackNameToEnv(stackName: string): void {
  // Path to .env file in the workspace root (parent of infrastructure directory)
  const envPath = path.join(__dirname, "..", "..", ".env");

  let envContent = "";

  // Read existing .env file if it exists
  if (fs.existsSync(envPath)) {
    envContent = fs.readFileSync(envPath, "utf8");
  }

  // Check if CDK_STACK_NAME already exists in the file
  const lines = envContent.split("\n");
  let stackNameExists = false;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("CDK_STACK_NAME=")) {
      lines[i] = `CDK_STACK_NAME=${stackName}`;
      stackNameExists = true;
      break;
    }
  }

  // If CDK_STACK_NAME doesn't exist, add it
  if (!stackNameExists) {
    // Add a comment and the variable
    if (envContent && !envContent.endsWith("\n")) {
      lines.push("");
    }
    lines.push("# CDK Stack Name (auto-generated)");
    lines.push(`CDK_STACK_NAME=${stackName}`);
  }

  // Write back to .env file
  fs.writeFileSync(envPath, lines.join("\n"));
  console.log(`Updated .env file with CDK_STACK_NAME=${stackName}`);
}

const app = new cdk.App();
const stackName = getStackName();

// Write the stack name to .env file for persistence and VS Code integration
writeStackNameToEnv(stackName);

const stack = new AgentCoreIntegrationStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});

// Apply CDK Nag AwsSolutionsChecks
// CDK Nag is disabled by default. Set CDK_NAG_ENABLED=true to enable
const cdkNagEnabled = process.env.CDK_NAG_ENABLED === "true";
if (cdkNagEnabled) {
  // Apply CDK Nag suppressions before running checks
  CdkNagSuppressions.applySuppressions(stack);

  cdk.Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));
  console.log(
    "CDK Nag AwsSolutionsChecks applied to the stack with suppressions"
  );
} else {
  console.log("CDK Nag disabled (set CDK_NAG_ENABLED=true to enable)");
}

// Output the stack name for reference
console.log(`Deploying stack: ${stackName}`);
