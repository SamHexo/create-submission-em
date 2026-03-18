#!/usr/bin/env python3
"""
Local Agent Testing Script

This script allows you to test your custom agent locally without deploying it.

Two modes of operation:
1. YAML Config Mode (recommended): Create test_config.yaml and run the script
2. Interactive Mode: Run without config file to be prompted for inputs

All webhooks are saved to local JSON files in the webhooks/ directory for inspection.

Usage:
    # Copy the example config
    cp test_config.yaml.example test_config.yaml

    # Edit test_config.yaml with your values

    # Run the test
    python test_agent_locally.py

    # Or run interactively (without config file)
    python test_agent_locally.py --interactive
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed")
    print("Install it with: pip install pyyaml")
    sys.exit(1)

# Import your custom agent
try:
    from custom_agent import create_agent
except ImportError:
    print("Error: Could not import create_agent from custom_agent.py")
    print("Make sure you have implemented your custom agent in custom_agent.py")
    sys.exit(1)


def prompt_user(message: str, default: str = None) -> str:
    """Prompt user for input with optional default."""
    if default:
        response = input(f"{message} [{default}]: ").strip()
        return response if response else default
    else:
        while True:
            response = input(f"{message}: ").strip()
            if response:
                return response
            print("This field is required. Please provide a value.")


def prompt_multiline(message: str) -> str:
    """Prompt user for multiline input."""
    print(f"{message}")
    print("(Press Ctrl+D or Ctrl+Z when done, or enter '---' on a new line)")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "---":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def prompt_json(message: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Prompt user for JSON input."""
    default_str = json.dumps(default) if default else "{}"
    print(f"{message}")
    print(f"Enter as JSON (default: {default_str})")
    print("Press Enter to use default, or paste JSON:")

    response = input().strip()
    if not response:
        return default or {}

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        print("Using default config instead.")
        return default or {}


def load_yaml_config(config_path: Path = Path("test_config.yaml")) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Uses same pattern as agent_api.py - reads UPPERCASE keys (matching production)
    with fallback to lowercase keys for backward compatibility.
    """
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Get config from YAML (same pattern as agent_api.py reads from env vars)
        problem_statement = raw_config.get("TASK_DESCRIPTION") or raw_config.get(
            "problem_statement", ""
        )
        max_steps = raw_config.get("MAX_ITERATIONS") or raw_config.get("max_steps", 10)
        experiment_id = raw_config.get("EXPERIMENT_ID") or raw_config.get(
            "experiment_id", "local-test-001"
        )
        project_id = raw_config.get("PROJECT_ID") or raw_config.get(
            "project_id", "local-project"
        )
        agent_config = raw_config.get("AGENT_CONFIG") or raw_config.get(
            "agent_config", {}
        )

        # Validate required fields
        if not problem_statement:
            print(
                f"Error: Missing TASK_DESCRIPTION or problem_statement in {config_path}"
            )
            sys.exit(1)

        if not max_steps:
            print(f"Error: Missing MAX_ITERATIONS or max_steps in {config_path}")
            sys.exit(1)

        # Parse API keys (prefer UPPERCASE individual keys, fallback to api_keys dict)
        api_keys = {
            "ANTHROPIC_API_KEY": raw_config.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": raw_config.get("OPENAI_API_KEY", ""),
            "GEMINI_API_KEY": raw_config.get("GEMINI_API_KEY", ""),
        }

        # Merge with api_keys dict if provided (backward compatibility)
        if "api_keys" in raw_config and isinstance(raw_config["api_keys"], dict):
            api_keys.update(raw_config["api_keys"])

        # Get continue_messages if provided
        continue_messages = raw_config.get("continue_messages", [])

        # Return normalized config
        return {
            "problem_statement": problem_statement,
            "max_steps": max_steps,
            "experiment_id": experiment_id,
            "project_id": project_id,
            "agent_config": agent_config,
            "api_keys": api_keys,
            "continue_messages": continue_messages,
        }
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def clean_webhooks_directory():
    """Clean previous webhook files."""
    webhooks_dir = Path("webhooks")
    if webhooks_dir.exists():
        response = (
            input("Previous webhook files found. Delete them? (y/N): ").strip().lower()
        )
        if response == "y":
            import shutil

            shutil.rmtree(webhooks_dir)
            print("✓ Cleaned webhooks directory")


async def main():
    """Main testing function."""
    print("=" * 70)
    print("Local Agent Testing Script".center(70))
    print("=" * 70)
    print()

    # Check for command-line flags
    config_path = Path("test_config.yaml")
    use_interactive = "--interactive" in sys.argv
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

    if config_path.exists() and not use_interactive:
        print(f"📄 Found {config_path}, loading configuration...")
        config = load_yaml_config(config_path)
        print("✓ Configuration loaded successfully")
        print()

        # Extract values from YAML
        problem_statement = config["problem_statement"]
        max_steps = config["max_steps"]
        api_keys = config["api_keys"]
        agent_config = config.get("agent_config", {})
        experiment_id = config.get("experiment_id", "local-test-001")
        project_id = config.get("project_id", "local-project")
        continue_messages = config.get("continue_messages", [])

        # Auto-confirm when using YAML config (no need to ask)
        auto_confirm = True

    else:
        if not config_path.exists():
            print("💡 No test_config.yaml found, using interactive mode")
            print(
                "   (Copy test_config.yaml.example to test_config.yaml for easier testing)"
            )
        else:
            print("🔧 Using interactive mode (--interactive flag detected)")
        print()

        print("Please provide the following information to test your agent:\n")

        # 1. Problem Statement
        print("1. Problem Statement")
        print("-" * 70)
        problem_statement = prompt_multiline(
            "Enter the problem statement for your agent"
        )
        print()

        # 2. Max Steps
        print("2. Maximum Steps")
        print("-" * 70)
        max_steps = int(prompt_user("Enter max steps", default="10"))
        print()

        # 3. API Keys (Optional - uses environment variables by default)
        print("3. API Keys (Optional)")
        print("-" * 70)
        print("API keys will be read from environment variables automatically.")
        print("Only provide keys here if you want to override environment values.")
        api_keys = {}

        # Check if environment has common API keys
        env_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
        if env_anthropic:
            print("✓ ANTHROPIC_API_KEY found in environment")

        # Optional: Override with custom keys
        override = prompt_user(
            "Override with custom API keys? (y/N)", default="n"
        ).lower()
        if override == "y":
            anthropic_key = prompt_user(
                "ANTHROPIC_API_KEY",
                default=env_anthropic,
            )
            if anthropic_key:
                api_keys["ANTHROPIC_API_KEY"] = anthropic_key

            # Add more API keys if needed
            add_more = prompt_user("Add more API keys? (y/N)", default="n").lower()
            while add_more == "y":
                key_name = prompt_user("API key name")
                key_value = prompt_user(f"{key_name} value")
                api_keys[key_name] = key_value
                add_more = prompt_user("Add another? (y/N)", default="n").lower()

        print()

        # 4. Agent Config
        print("4. Agent Configuration")
        print("-" * 70)
        agent_config = prompt_json(
            "Enter agent config (optional)",
            default={
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 1.0,
                "max_tokens": 4096,
            },
        )
        print()

        # 5. Experiment and Project IDs
        print("5. Test Metadata")
        print("-" * 70)
        experiment_id = prompt_user("Experiment ID", default="local-test-001")
        project_id = prompt_user("Project ID", default="local-project")
        print()

        # No continue messages in interactive mode (will prompt after start)
        continue_messages = []

    # Clean previous webhooks
    clean_webhooks_directory()

    # Summary
    print("=" * 70)
    print("Configuration Summary".center(70))
    print("=" * 70)
    print(f"Problem Statement: {problem_statement[:60]}...")
    print(f"Max Steps: {max_steps}")
    print(f"API Keys: {', '.join(api_keys.keys())}")
    print(f"Agent Config: {json.dumps(agent_config, indent=2)}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Project ID: {project_id}")
    print()

    # Confirm (skip if using YAML config or --yes flag)
    if not auto_confirm:
        confirm = prompt_user("Start agent? (Y/n)", default="y").lower()
        if confirm != "y" and confirm != "":
            print("Aborted.")
            return
    else:
        print("✓ Auto-confirmed (using YAML config)")

    print()
    print("=" * 70)
    print("Starting Agent Execution".center(70))
    print("=" * 70)
    print()
    print("NOTE: Webhooks will be saved to ./webhooks/ directory")
    print("      Check the JSON files to see agent progress and data")
    print()

    # Create agent instance (NO webhook_url means it saves locally)
    try:
        agent = create_agent(
            experiment_id=experiment_id,
            project_id=project_id,
            problem_statement=problem_statement,
            max_steps=max_steps,
            api_keys=api_keys,
            webhook_url=None,  # This makes it save locally
            agent_config=agent_config,
            jwt_token=None,
        )

        # Start agent (skip if we have continue_messages)
        if continue_messages:
            print("⏭️  Skipping start() - testing continue_agent only\n")
            print("⚠️  WARNING: Skipping start() means agent state must already exist!")
            print("   This requires either:")
            print("   1. Agent container is already running with existing state")
            print("   2. You are reusing workspace from a previous run\n")
        else:
            print("🚀 Agent starting...\n")
            await agent.start()

            print()
            print("=" * 70)
            print("Agent Execution Completed".center(70))
            print("=" * 70)
            print()
            print("✓ Agent finished successfully")

        # Test continue_agent
        # Either from YAML config or interactive prompts
        if continue_messages:
            # Process continue messages from YAML config
            print()
            print("=" * 70)
            print(f"Testing {len(continue_messages)} continue message(s)".center(70))
            print("=" * 70)

            for i, msg_config in enumerate(continue_messages, 1):
                print()
                print(f"--- Continue Message {i} ---")
                user_message = msg_config.get("message", "")
                new_max_steps = msg_config.get("new_max_steps", max_steps + 5)
                step_number = msg_config.get("step_number")
                # branch_name is internal (fetched from DB in production), default to None
                branch_name = msg_config.get("branch_name", None)

                print(f"📨 Message: {user_message[:60]}...")
                print(f"   Max steps: {new_max_steps}")
                if step_number is not None:
                    print(f"   From step: {step_number}")
                if branch_name:
                    print(f"   Branch: {branch_name}")
                print()

                await agent.continue_agent(
                    user_message=user_message,
                    new_max_steps=new_max_steps,
                    step_number=step_number,
                    branch_name=branch_name,
                )

                print(f"✓ Continue message {i} completed")

        elif not auto_confirm:
            # Interactive mode: ask if user wants to test continue
            print()
            test_continue = prompt_user(
                "Test continue_agent with a message? (y/N)", default="n"
            ).lower()

            if test_continue == "y":
                print()
                print("=" * 70)
                print("Testing continue_agent".center(70))
                print("=" * 70)
                print()

                user_message = prompt_multiline("Enter message to send to agent")
                new_max_steps = int(
                    prompt_user("New max steps", default=str(max_steps + 5))
                )

                # Optional: step number to continue from
                step_input = prompt_user(
                    "Step number to continue from (leave empty for current)", default=""
                )
                step_number = int(step_input) if step_input else None

                # branch_name defaults to None (in production, Django fetches from DB)
                branch_name = None

                print()
                print("📨 Sending message to agent...")
                if step_number is not None:
                    print(f"   From step: {step_number}")
                print()

                await agent.continue_agent(
                    user_message=user_message,
                    new_max_steps=new_max_steps,
                    step_number=step_number,
                    branch_name=branch_name,
                )

                print()
                print("✓ Continue agent test completed")

        print()
        print("Check the webhooks/ directory to see all the agent events:")

        webhooks_dir = Path("webhooks")
        if webhooks_dir.exists():
            webhook_files = sorted(webhooks_dir.glob("*.json"))
            if webhook_files:
                print(f"\nGenerated {len(webhook_files)} webhook files:")
                for filepath in webhook_files:
                    print(f"  - {filepath}")
            else:
                print("\nNo webhook files were generated.")
        else:
            print("\nNo webhooks directory found.")

    except Exception as e:
        print()
        print("=" * 70)
        print("Agent Execution Failed".center(70))
        print("=" * 70)
        print()
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
