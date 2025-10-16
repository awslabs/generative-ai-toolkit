#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { AgentCoreIntegrationStack } from "../lib/agentcore-stack";

const app = new cdk.App();
new AgentCoreIntegrationStack(app, "AgentCoreStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
