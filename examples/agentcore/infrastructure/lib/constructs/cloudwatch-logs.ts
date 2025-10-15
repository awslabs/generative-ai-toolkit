import * as cdk from "aws-cdk-lib";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

export interface CloudWatchLogsProps {
  agentLogGroupName: string;
}

export class CloudWatchLogs extends Construct {
  public readonly agentLogGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props: CloudWatchLogsProps) {
    super(scope, id);

    // CloudWatch Log Group for Agent
    this.agentLogGroup = new logs.LogGroup(this, "AgentLogGroup", {
      logGroupName: props.agentLogGroupName,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      retention: logs.RetentionDays.ONE_WEEK,
    });
  }
}
