"""
Agent Logic Base Class

Simple standard interface for building self-contained agents.
Users extend this class and implement: start, continue_agent, abort
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from models import ActionMessage, ObservationMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """

    Base class for self-contained agent logic.

    Standard interface:
    - start(): Begin agent execution
    - continue_agent(): Continue with user input
    - abort(): Stop execution

    Agent loop webhooks:
    - STEP_CREATED: Step is created
    - ACTION_RECEIVED: Action received from orchestrator
    - OBSERVATION_RECEIVED: Observation received after action execution
    - STEP_FINISHED: Step finished with commit info
    """

    def __init__(
        self,
        experiment_id: str,
        project_id: str,
        problem_statement: str,
        max_steps: int,
        api_keys: Dict[str, str],
        webhook_url: Optional[str] = None,
        agent_config: Dict[str, Any] = None,
        jwt_token: str = None,
    ):
        """Initialize agent.

        Args:
            experiment_id: Unique experiment identifier
            project_id: Project identifier
            problem_statement: Task description for the agent
            max_steps: Maximum number of steps allowed
            api_keys: Dictionary of API keys (e.g., {"ANTHROPIC_API_KEY": "..."})
                     If empty, will automatically use environment variables
            webhook_url: Optional webhook URL. If None, webhooks are saved locally
            agent_config: Optional agent configuration dictionary
            jwt_token: Optional JWT token for authentication
        """
        self.experiment_id = experiment_id
        self.project_id = project_id
        self.webhook_url = webhook_url
        self.problem_statement = problem_statement
        self.max_steps = max_steps

        # Merge provided api_keys with environment variables (provided keys take precedence)
        self.api_keys = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        }
        # Override with explicitly provided keys
        self.api_keys.update(api_keys or {})

        self.agent_config = agent_config or {}
        self.jwt_token = jwt_token
        self.current_step = 0
        self.is_aborted = False

    @abstractmethod
    async def start(self) -> None:
        """
        Start agent execution.

        Called when experiment starts.
        Implement your main agent loop here.
        """
        pass

    @abstractmethod
    async def continue_agent(
        self,
        user_message: str,
        new_max_steps: int,
        step_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> None:
        """
        Continue execution with user input.

        Called when user sends a message to continue.

        Args:
            user_message: Message from user
            new_max_steps: New maximum steps limit
            step_number: Optional step number to continue from
            branch_name: Optional git branch name
        """
        pass

    @abstractmethod
    async def abort(self) -> None:
        """
        Abort execution and clean up.

        Called when user requests to stop.
        """
        pass

    # Helper methods

    async def send_webhook(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Send webhook to core app or save locally if no webhook URL.

        Args:
            event_type: Event type (STEP_CREATED, ACTION_RECEIVED, OBSERVATION_RECEIVED, STEP_FINISHED)
            data: Event data
        """
        payload = {
            "event_type": event_type,
            "task_id": self.experiment_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **data,
        }

        # If no webhook URL, save locally for testing
        if not self.webhook_url:
            self._save_webhook_locally(event_type, payload, data.get("step_number"))
            return

        try:
            headers = {}
            if self.jwt_token:
                headers["Authorization"] = f"Bearer {self.jwt_token}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.webhook_url, json=payload, headers=headers
                )
                response.raise_for_status()
        except Exception as e:
            print(f"Failed to send webhook {event_type}: {e}")

    def _save_webhook_locally(
        self, event_type: str, payload: Dict[str, Any], step_number: Optional[int]
    ) -> None:
        """
        Save webhook payload to local JSON file for testing.

        Args:
            event_type: Event type
            payload: Webhook payload
            step_number: Optional step number
        """
        # Create webhooks directory if it doesn't exist
        webhooks_dir = Path("webhooks")
        webhooks_dir.mkdir(exist_ok=True)

        # Generate filename based on event type and step
        if event_type == "INITIAL_MESSAGES":
            filename = "00_initial_messages.json"
        elif event_type == "ACTION_RECEIVED":
            filename = f"{step_number:02d}_action_received.json"
        elif event_type == "STEP_FINISHED":
            filename = f"{step_number:02d}_step_finished.json"
        elif event_type == "EXPERIMENT_COMPLETED":
            result_step = step_number if step_number is not None else "final"
            filename = f"result_{result_step}.json"
        elif event_type == "EXPERIMENT_FAILED":
            filename = f"failed_{step_number or 'unknown'}.json"
        elif event_type == "EXPERIMENT_ABORTED":
            filename = "aborted.json"
        else:
            filename = f"{event_type.lower()}_{step_number or 'unknown'}.json"

        filepath = webhooks_dir / filename

        # Save payload as JSON
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"✓ Saved webhook locally: {filepath}")

    async def send_action_received(
        self,
        step_number: int,
        action_id: str,
        action_type: str,
        action_message: Dict[str, Any],
    ) -> None:
        """
        Send ACTION_RECEIVED webhook when LLM generates an action.

        This is called BEFORE executing the action. It creates the step and stores
        the LLM's complete response (role, content, tool_calls, completion_details).

        Args:
            step_number: Step number
            action_id: Unique action identifier
            action_type: Type of action (e.g., "execute_bash", "finish")
            action_message: Complete LLM response with structure:
                {
                    "role": "assistant",
                    "content": "thought/reasoning text",
                    "tool_calls": [...],  # Optional
                    "completion_details": {...}  # Optional, contains usage/cost
                }
        """
        # Validate with Pydantic
        validated_message = ActionMessage.create(
            content=action_message.get("content", ""),
            action_id=action_id,
            action_type=action_type,
            tool_calls=action_message.get("tool_calls"),
            completion_details=action_message.get("completion_details"),
        )

        # Build webhook data
        webhook_data = {
            "step_number": step_number,
            "action_id": action_id,
            "action_type": action_type,
            "action_message": validated_message.model_dump(by_alias=True),
        }

        await self.send_webhook("ACTION_RECEIVED", webhook_data)

    def _get_or_create_commit(self, workspace_dir: str = "/workspace") -> Optional[str]:
        """
        Get current commit ID or create a new commit if there are changes.

        Excludes files larger than 500MB. Media files (images, videos, audio) are
        excluded via .gitignore.

        Args:
            workspace_dir: Path to workspace directory (default: /workspace)

        Returns:
            Commit ID (SHA) or None if git is not initialized
        """
        import os
        import subprocess

        try:
            # Check if git repo exists
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.info("Git repository not initialized in workspace")
                return None

            # Check if there are any changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if status_result.stdout.strip():
                # There are changes, create a commit
                logger.info("Changes detected in workspace, creating auto-commit...")

                # Find and exclude very large files (>500MB) before staging
                MAX_FILE_SIZE_MB = 500
                exclude_path = os.path.join(workspace_dir, ".git", "info", "exclude")

                try:
                    # Use find to identify large files (fast even with 100k+ files)
                    find_result = subprocess.run(
                        [
                            "find",
                            ".",
                            "-type",
                            "f",
                            "-size",
                            f"+{MAX_FILE_SIZE_MB}M",
                            "-not",
                            "-path",
                            "./.git/*",
                        ],
                        cwd=workspace_dir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    large_files = [
                        f.lstrip("./")
                        for f in find_result.stdout.strip().split("\n")
                        if f
                    ]

                    if large_files:
                        logger.warning(
                            f"Found {len(large_files)} file(s) >500MB, adding to git exclude"
                        )

                        # Read existing patterns
                        existing_patterns = set()
                        if os.path.exists(exclude_path):
                            with open(exclude_path, "r") as f:
                                existing_patterns = {
                                    line.strip()
                                    for line in f
                                    if line.strip() and not line.startswith("#")
                                }

                        # Add new large files
                        new_patterns = [
                            f for f in large_files if f not in existing_patterns
                        ]
                        if new_patterns:
                            with open(exclude_path, "a") as f:
                                f.write("\n# Large files excluded from auto-commit\n")
                                for pattern in new_patterns:
                                    f.write(f"{pattern}\n")

                            for large_file in large_files[:5]:
                                logger.warning(f"  - {large_file}")
                            if len(large_files) > 5:
                                logger.warning(f"  ... and {len(large_files) - 5} more")

                except Exception as e:
                    logger.warning(f"Could not check for large files: {e}")

                # Stage all changes (gitignore and git/info/exclude will filter)
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=workspace_dir,
                    capture_output=True,
                    timeout=60,
                )

                # Create commit with timestamp
                from datetime import datetime

                commit_message = (
                    f"Auto-commit at step {self.current_step}\n\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}Z"
                )

                commit_result = subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    cwd=workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if commit_result.returncode != 0:
                    if "nothing to commit" in commit_result.stdout.lower():
                        logger.info("No files to commit after filtering")
                    else:
                        logger.warning(
                            f"Failed to create commit: {commit_result.stderr}"
                        )
                else:
                    logger.info("Auto-commit created successfully")

            # Get current commit ID
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if commit_result.returncode == 0:
                commit_id = commit_result.stdout.strip()
                logger.info(f"Current commit ID: {commit_id[:8]}")
                return commit_id
            else:
                logger.warning("Failed to get commit ID")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("Git command timed out")
            return None
        except Exception as e:
            logger.warning(f"Error getting/creating commit: {e}")
            return None

    async def send_step_finished(
        self,
        step_number: int,
        action_id: Optional[str] = None,
        action_type: Optional[str] = None,
        observation_content: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        commit_id: Optional[str] = None,
        git_branch: Optional[str] = None,
        error: Optional[str] = None,
        user_message: Optional[str] = None,
        action_message: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send STEP_FINISHED webhook after executing an action.

        This is called AFTER action execution completes. It updates the step
        with the observation result and git commit info.

        For user continue messages, only step_number and user_message are needed.

        Args:
            step_number: Step number (same as ACTION_RECEIVED)
            action_id: Unique action ID (from ACTION_RECEIVED) - auto-generated if None
            action_type: Type of action (from ACTION_RECEIVED)
            observation_content: Execution result content
            tool_call_id: Tool call ID to link observation to action
            commit_id: Optional git commit ID (auto-commits if None and changes exist)
            git_branch: Optional git branch name
            error: Optional error message if execution failed
            user_message: Optional user message content (for continue requests)
            action_message: Optional action message dict (for self-contained agents)
        """
        # Auto-commit if commit_id not provided
        if commit_id is None:
            commit_id = self._get_or_create_commit()

        webhook_data = {
            "step_number": step_number,
            "commit_id": commit_id,
            "git_branch": git_branch,
            "error": error,
        }

        # For user continue messages
        if user_message:
            from datetime import datetime

            webhook_data["input_text"] = user_message
            webhook_data["user_message"] = {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        # For normal action execution results
        elif observation_content is not None:
            # Auto-generate missing IDs for self-contained agents
            # These defaults ensure the webhook always has valid values
            if not action_id:
                action_id = f"action_{self.experiment_id}_{step_number}"
            if not action_type:
                action_type = "assistant"
            if not tool_call_id:
                tool_call_id = f"call_{step_number}"

            # Create ObservationMessage with Pydantic validation
            observation_message = ObservationMessage.create(
                name=action_type,
                content=observation_content,
                tool_call_id=tool_call_id,
                action_id=action_id,
                error=error,
            )
            webhook_data["observation_message"] = observation_message.model_dump(
                by_alias=True
            )

            # Add action_message if provided (for self-contained agents)
            if action_message:
                webhook_data["action_message"] = action_message
        else:
            raise ValueError(
                f"Must provide either user_message OR observation_content. "
                f"Got: user_message={user_message}, observation_content={observation_content}"
            )

        await self.send_webhook("STEP_FINISHED", webhook_data)

    async def send_initial_messages(
        self,
        system_message: str,
        user_message: str,
        step_number: int = 0,
    ) -> None:
        """
        Send INITIAL_MESSAGES webhook when agent starts.

        This webhook signals that the agent is ready and provides the initial
        system and user messages that set up the agent's context.

        Args:
            system_message: System prompt/instructions for the agent
            user_message: Initial user message with task description
            step_number: Step number (usually 0 for initial messages)
        """
        webhook_data = {
            "step_number": step_number,
            "system_message": {
                "role": "system",
                "content": system_message,
            },
            "user_message": {
                "role": "user",
                "content": user_message,
            },
        }

        await self.send_webhook("INITIAL_MESSAGES", webhook_data)

    async def send_iteration_result(
        self,
        success: bool = True,
        summary: str = "",
        score: Optional[float] = None,
        approach: str = "",
        commit_id: Optional[str] = None,
        git_branch: Optional[str] = None,
        suggested_improvements: Optional[str] = None,
        step_number: Optional[int] = None,
    ) -> None:
        """
        Save an intermediate iteration result WITHOUT marking experiment as completed.

        Use this to save checkpoints/iteration results while the agent continues running.
        This creates a result entry in the database but keeps the experiment in RUNNING state,
        allowing you to continue with more steps.

        For the final result that marks the experiment as done, use send_experiment_completed().

        Args:
            success: Whether iteration completed successfully
            summary: Summary of what was accomplished in this iteration
            score: Optional numeric score/metric (0-100)
            approach: Description of approach taken
            commit_id: Git commit ID of current state (auto-commits if None and changes exist)
            git_branch: Git branch name
            suggested_improvements: Optional suggestions for improvement
            step_number: Optional step number that triggered this result save
        """
        # Auto-commit if commit_id not provided
        if commit_id is None:
            commit_id = self._get_or_create_commit()

        # Sanitize score - replace inf/nan with None to prevent JSON serialization errors
        # Database constraint: DecimalField(max_digits=8, decimal_places=3)
        # Valid range: -99999.999 to 99999.999
        import math

        if score is not None:
            if math.isinf(score) or math.isnan(score):
                print(
                    f"⚠️  Warning: Invalid score value {score} at step {step_number}, setting to None"
                )
                score = None
            elif score < -99999.999 or score > 99999.999:
                print(
                    f"⚠️  Warning: Score {score} out of database range [-99999.999, 99999.999] "
                    f"at step {step_number}, setting to None"
                )
                score = None

        webhook_data = {
            "success": success,
            "summary": summary,
            "score": score,
            "approach": approach,
            "commit_id": commit_id,
            "git_branch": git_branch,
            "suggested_improvements": suggested_improvements,
            "step_number": step_number,
        }

        await self.send_webhook("ITERATION_RESULT", webhook_data)

    async def send_experiment_completed(
        self,
        success: bool = True,
        summary: str = "",
        score: Optional[float] = None,
        approach: str = "",
        commit_id: Optional[str] = None,
        git_branch: Optional[str] = None,
        suggested_improvements: Optional[str] = None,
        step_number: Optional[int] = None,
    ) -> None:
        """
        Send EXPERIMENT_COMPLETED webhook to save final result and mark experiment as completed.

        This saves the experiment result to the database AND marks the experiment as COMPLETED,
        which prevents further ACTION_RECEIVED/STEP_FINISHED webhooks from being processed.

        Use send_iteration_result() for intermediate results, and this method only once at the end.

        Args:
            success: Whether experiment completed successfully
            summary: Summary of what was accomplished
            score: Optional numeric score/metric (0-100)
            approach: Description of approach taken
            commit_id: Git commit ID of current state
            git_branch: Git branch name
            suggested_improvements: Optional suggestions for improvement
            step_number: Optional step number that triggered this result save
        """
        # Sanitize score - replace inf/nan with None to prevent JSON serialization errors
        # Database constraint: DecimalField(max_digits=8, decimal_places=3)
        # Valid range: -99999.999 to 99999.999
        import math

        if score is not None:
            if math.isinf(score) or math.isnan(score):
                print(
                    f"⚠️  Warning: Invalid score value {score} at step {step_number}, setting to None"
                )
                score = None
            elif score < -99999.999 or score > 99999.999:
                print(
                    f"⚠️  Warning: Score {score} out of database range [-99999.999, 99999.999] "
                    f"at step {step_number}, setting to None"
                )
                score = None

        webhook_data = {
            "success": success,
            "summary": summary,
            "score": score,
            "approach": approach,
            "commit_id": commit_id,
            "git_branch": git_branch,
            "suggested_improvements": suggested_improvements,
            "step_number": step_number,
        }

        await self.send_webhook("EXPERIMENT_COMPLETED", webhook_data)

    async def send_experiment_failed(
        self,
        error_message: str,
        step_number: Optional[int] = None,
        commit_id: Optional[str] = None,
        git_branch: Optional[str] = None,
    ) -> None:
        """
        Send EXPERIMENT_FAILED webhook when experiment fails.

        Args:
            error_message: Description of the failure
            step_number: Optional step number where failure occurred
            commit_id: Optional git commit ID
            git_branch: Optional git branch name
        """
        webhook_data = {
            "error": error_message,
            "step_number": step_number,
            "commit_id": commit_id,
            "git_branch": git_branch,
        }

        await self.send_webhook("EXPERIMENT_FAILED", webhook_data)

    async def send_experiment_aborted(
        self,
        reason: str = "Aborted by user",
        last_step: Optional[int] = None,
    ) -> None:
        """
        Send EXPERIMENT_ABORTED webhook when experiment is aborted.

        Args:
            reason: Reason for abortion
            last_step: Last completed step number
        """
        webhook_data = {
            "reason": reason,
            "last_step": last_step or self.current_step,
        }

        await self.send_webhook("EXPERIMENT_ABORTED", webhook_data)
