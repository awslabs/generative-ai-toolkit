# AgentCore CDK Deployment Guide

## Stack Naming

Each developer should deploy with a unique stack name to avoid conflicts. The system supports two methods and automatically saves the stack name to a `.env` file for persistence:

### Method 1: Environment Variable (Recommended for CI/CD)

Set the `CDK_STACK_NAME` environment variable:

```bash
export CDK_STACK_NAME="your-name-agentcore-stack"
cdk deploy
```

Or inline:
```bash
CDK_STACK_NAME="alice-agentcore-stack" cdk deploy
```

### Method 2: Auto-naming (Simplest for Development)

Just run `cdk deploy` without setting any environment variables. The system will automatically use your username:

```bash
cdk deploy
```

This will create a stack named: `{your-username}-agentcore-stack`

### Automatic .env File Management

The system automatically writes the determined stack name to `examples/agentcore/.env` as `CDK_STACK_NAME=your-stack-name`. This:
- Makes the stack name available to VS Code's Python extension
- Persists the stack name for future deployments
- Helps with consistency across development tools

## Deployment Steps

1. **Navigate to infrastructure directory:**
   ```bash
   cd examples/agentcore/infrastructure
   ```

2. **Install dependencies (first time only):**
   ```bash
   npm install
   ```

3. **Check what stack name will be used (optional):**
   ```bash
   npm run check-stack-name
   ```

4. **Deploy the stack:**
   ```bash
   # Option A: Auto-naming (uses your username)
   npm run deploy-with-outputs
   
   # Option B: Custom name
   CDK_STACK_NAME="my-custom-stack-name" npm run deploy-with-outputs
   
   # Option C: Without outputs file
   npm run deploy
   ```

4. **Verify deployment:**
   The deployment will output the stack name being used:
   ```
   Deploying stack: alice-agentcore-stack
   ```

## Outputs File

After deployment, if you used the `--outputs-file` flag, your configuration outputs will be saved to:
```
examples/agentcore/infrastructure/cdk-outputs.json
```

This file contains the Cognito User Pool ID, Client ID, and other configuration values needed by the agent.

## Cleanup

To destroy your stack:
```bash
# If using auto-naming
cdk destroy

# If using custom name
CDK_STACK_NAME="my-custom-stack-name" cdk destroy
```

## Examples

### Developer Alice
```bash
cd examples/agentcore/infrastructure
npm run check-stack-name  # Shows: alice-agentcore-stack
npm run deploy-with-outputs
# Creates: alice-agentcore-stack (assuming Alice's username is 'alice')
```

### Developer Bob with Custom Name
```bash
cd examples/agentcore/infrastructure
CDK_STACK_NAME="bob-test-stack" npm run check-stack-name  # Shows: bob-test-stack
CDK_STACK_NAME="bob-test-stack" npm run deploy-with-outputs
# Creates: bob-test-stack
```

### CI/CD Pipeline
```bash
export CDK_STACK_NAME="prod-agentcore-stack"
cd examples/agentcore/infrastructure
npm run check-stack-name  # Shows: prod-agentcore-stack
npm run deploy-with-outputs
# Creates: prod-agentcore-stack
```

## Troubleshooting

### Stack Already Exists
If you get a "stack already exists" error, either:
1. Use a different stack name
2. Or destroy the existing stack first: `cdk destroy`

### Username Not Detected
If the auto-naming shows "dev-agentcore-stack", it means your username environment variable isn't set. You can:
1. Set it manually: `export USER=yourname`
2. Or use the explicit naming method: `CDK_STACK_NAME="yourname-agentcore-stack"`