# CreateSumbissionAgentEm - Custom BYOA Agent

## Description

A custom agent implementation for the Emily BYOA (Bring Your Own Agent) platform.

## Quick Start

### 1. Implement Your Agent

Edit `custom_agent.py` and implement the three required methods:

- `start()`: Main agent loop
- `continue_agent()`: Handle user feedback
- `abort()`: Clean up and stop

### 2. Install Dependencies Locally (for testing)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Test Locally

#### Option A: Interactive Testing Script (Recommended)

Use the provided interactive testing script to test your agent without any webhook setup:

```bash
# Run the interactive test script
python test_agent_locally.py
```

This script will:
- Prompt you for all necessary inputs (problem statement, max steps, optional API keys, agent config)
- Use environment variables for API keys automatically (you can override if needed)
- Run your agent locally without needing a webhook server
- Save all webhook events as JSON files in `webhooks/` directory
- Show you exactly what data would be sent to the platform

Perfect for rapid development and debugging!

**Example workflow:**
```bash
$ python test_agent_locally.py
======================================================================
                    Local Agent Testing Script
======================================================================

Please provide the following information to test your agent:

1. Problem Statement
----------------------------------------------------------------------
Enter the problem statement for your agent
(Press Ctrl+D or Ctrl+Z when done, or enter '---' on a new line)
Fix the bug in src/main.py
---

2. Maximum Steps
----------------------------------------------------------------------
Enter max steps [10]: 5

3. API Keys (Optional)
----------------------------------------------------------------------
API keys will be read from environment variables automatically.
Only provide keys here if you want to override environment values.
✓ ANTHROPIC_API_KEY found in environment
Override with custom API keys? (y/N) [n]:

4. Agent Configuration
----------------------------------------------------------------------
Enter agent config (optional)
Enter as JSON (default: {"model": "claude-3-5-sonnet-20241022", ...})
Press Enter to use default, or paste JSON:

...

🚀 Agent starting...

✓ Saved webhook locally: webhooks/00_initial_messages.json
✓ Saved webhook locally: webhooks/01_action_received.json
✓ Saved webhook locally: webhooks/01_step_finished.json
...

✓ Agent finished successfully

Check the webhooks/ directory to see all the agent events:
  - webhooks/00_initial_messages.json
  - webhooks/01_action_received.json
  - webhooks/01_step_finished.json
  - webhooks/result_final.json
```

#### Option B: Manual Testing

Or test manually with a Python script:

```python
import asyncio
from custom_agent import create_agent

async def test():
    agent = create_agent(
        experiment_id="test-123",
        project_id="proj-456",
        webhook_url=None,  # Set to None for local testing
        problem_statement="Test task",
        max_steps=10,
        api_keys={"ANTHROPIC_API_KEY": "sk-..."},  # pragma: allowlist secret
        agent_config={},
        jwt_token=None,
    )
    await agent.start()

asyncio.run(test())
```

**Note:** When `webhook_url` is `None`, all webhook events are saved to `webhooks/*.json` files locally.

### 4. Build Docker Image

Use the Emily CLI to build your agent:

```bash
# From the parent directory
emily custom-agent build create-sumbission-agent-em

# Or with custom tag
emily custom-agent build create-sumbission-agent-em -t my-custom-tag:v1.0

# Or with custom base image
emily custom-agent build create-sumbission-agent-em --base-image emily/byoa-base:v2.0
```

This will:
- Generate a Dockerfile automatically
- Use the `emily/byoa-base:latest` base image
- Copy all your files to `/app/`
- Install any additional requirements
- Build and tag the image

### 5. Run Your Agent

```bash
docker run -it emily/create-sumbission-agent-em:latest
```

## Agent Structure

```
create-sumbission-agent-em/
├── base_agent.py      # Base class with webhook helpers (from Emily CLI)
├── models.py          # Pydantic models for messages (from Emily CLI)
├── custom_agent.py    # YOUR IMPLEMENTATION
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore patterns for workspace
└── README.md         # This file
```

**Note:** No Dockerfile needed! The Emily CLI generates it for you.

## Project Files

### `.gitignore` - Workspace Git Configuration

The `.gitignore` file in your agent project controls what gets excluded from git commits **in the workspace** during agent execution.

**How it works:**
- When your agent container starts, it **copies** `.gitignore` to `/workspace/.gitignore`
- This gives you full control over what files are committed during agent execution
- The default `.gitignore` excludes:
  - ML model files (`.pth`, `.pt`, `.ckpt`, `.safetensors`, etc.)
  - Media files (images, videos, audio)
  - Binary data (`.npy`, `.npz`, `.pkl`)
  - Data files (`.parquet`)
  - Python cache files (`__pycache__`, `.pyc`)
  - Virtual environments (`.venv`, `venv/`)

**Customize it:**
```bash
# Edit .gitignore in your agent project
nano .gitignore

# Your changes will automatically be used in the workspace
# when the container starts
```

**Why this matters:**
- Prevents huge model files from being committed to git
- Keeps workspace size manageable
- Avoids OOM errors from git operations on large files
- You control what gets tracked vs. ignored

## Webhook Protocol

Your agent communicates with the Emily platform using webhooks:

1. **INITIAL_MESSAGES**: Send system/user prompts at start
2. **ACTION_RECEIVED**: Before executing each action
3. **STEP_FINISHED**: After executing each action
4. **EXPERIMENT_COMPLETED**: Save results (can call multiple times)
5. **EXPERIMENT_FAILED**: Report failures
6. **EXPERIMENT_ABORTED**: Report abortion

See `base_agent.py` for detailed documentation.

### Local Testing Mode

When `webhook_url` is `None` or not provided, the agent operates in **local testing mode**:
- All webhook events are saved as JSON files in `webhooks/` directory
- Files are named descriptively: `01_action_received.json`, `01_step_finished.json`, etc.
- This allows you to inspect agent behavior without a webhook server
- Perfect for development and debugging

**Webhook file naming:**
- `00_initial_messages.json` - Initial system and user messages
- `{step}_action_received.json` - Action/tool calls from LLM
- `{step}_step_finished.json` - Observations/results after execution
- `result_{step}.json` - Experiment results (can be multiple)
- `failed_{step}.json` - Failure events
- `aborted.json` - Abortion events

## Available Attributes

- `self.experiment_id`: Unique experiment ID
- `self.project_id`: Project ID
- `self.problem_statement`: The task to solve
- `self.max_steps`: Maximum steps allowed
- `self.api_keys`: Dict of API keys (auto-populated from environment variables)
- `self.agent_config`: Custom configuration dict
- `self.jwt_token`: JWT for API authentication
- `self.current_step`: Current step number
- `self.is_aborted`: Boolean flag for abortion

## Action Types

Action types are flexible - use any string that describes your action:

- `"execute_bash"`: Execute shell command
- `"read_file"`: Read file contents
- `"create_file"`: Create new file
- `"your_custom_action"`: Your custom action

## Example Implementation

See `custom_agent.py` for a skeleton implementation with TODOs.

## Adding Dependencies

Add any Python dependencies to `requirements.txt`. They will be installed automatically when building the Docker image.

```
# requirements.txt
httpx==0.28.1
pydantic==2.10.5
openai==1.0.0       # Add your dependencies
anthropic==0.5.0    # Add your dependencies
```

## Base Image

The agent uses `emily/byoa-base:latest` which includes:

- Python 3.12 + CUDA 12.4.1 support
- System dependencies (git, curl, nginx, ssh)
- Common ML libraries (numpy, pandas, scikit-learn, etc.)
- Base agent infrastructure
- Workspace environment

## Next Steps

1. Implement your agent logic in `custom_agent.py`
2. Add any additional dependencies to `requirements.txt`
3. Test locally with a Python virtual environment
4. Build Docker image: `emily custom-agent build create-sumbission-agent-em`
5. Test Docker image: `docker run -it emily/create-sumbission-agent-em:latest`
6. Register with Emily platform

## Support

For questions or issues, refer to the Emily BYOA documentation.
