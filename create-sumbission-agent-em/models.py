"""
Message Interface Models

These models define the structure for action_message and observation_message
that match the backend's expected format for storing LLM interactions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ===== Tool Call Models =====


class ToolCallFunction(BaseModel):
    """Function details within a tool call"""

    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    """Individual tool call made by the LLM"""

    id: str
    type: str = "function"
    function: ToolCallFunction


# ===== Completion Details Models =====


class CompletionUsage(BaseModel):
    """Token usage information from LLM completion"""

    total_tokens: int
    cached_tokens: int = 0
    prompt_tokens: int
    completion_tokens: int


class CompletionDetails(BaseModel):
    """LLM completion metadata and usage information"""

    id: str
    model: str
    usage: CompletionUsage
    object: str = "chat.completion"
    created: int  # Unix timestamp
    completion_cost: Optional[float] = None


# ===== Metadata Models =====


class ActionMetadata(BaseModel):
    """Metadata for action messages"""

    action_id: str
    timestamp: str  # ISO 8601 format
    action_type: str


class ObservationMetadata(BaseModel):
    """Metadata for observation messages"""

    error: Optional[str] = None
    action_id: str
    timestamp: str  # ISO 8601 format


# ===== Main Message Models =====


class ActionMessage(BaseModel):
    """
    Action message from LLM API response.

    This represents what the LLM decided to do, including:
    - The LLM's reasoning/content
    - Tool calls it wants to make (optional)
    - Completion details like token usage (optional)
    """

    model_config = ConfigDict(populate_by_name=True)

    role: str = "assistant"
    content: str
    metadata: Optional[ActionMetadata] = Field(default=None, alias="_metadata")
    tool_calls: Optional[List[ToolCall]] = None
    completion_details: Optional[CompletionDetails] = None

    @classmethod
    def create(
        cls,
        content: str,
        action_id: str,
        action_type: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        completion_details: Optional[Dict[str, Any]] = None,
    ) -> "ActionMessage":
        """
        Helper to create an ActionMessage with automatic metadata.

        Args:
            content: The LLM's response content
            action_id: Unique action identifier
            action_type: Type of action being performed
            tool_calls: Optional list of tool calls (as dicts)
            completion_details: Optional completion metadata (as dict)

        Returns:
            ActionMessage instance
        """
        metadata = ActionMetadata(
            action_id=action_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            action_type=action_type,
        )

        # Convert tool_calls from dicts if provided
        tool_call_objs = None
        if tool_calls:
            tool_call_objs = [ToolCall(**tc) for tc in tool_calls]

        # Convert completion_details from dict if provided
        completion_obj = None
        if completion_details:
            completion_obj = CompletionDetails(**completion_details)

        return cls(
            role="assistant",
            content=content,
            metadata=metadata,
            tool_calls=tool_call_objs,
            completion_details=completion_obj,
        )


class ObservationMessage(BaseModel):
    """
    Observation message from tool/action execution.

    This represents the result of executing a tool/action, including:
    - The execution result content
    - Link back to which tool call this responds to
    - Error information if execution failed
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str  # Tool/function name
    role: str = "tool"
    content: str  # Execution result
    metadata: Optional[ObservationMetadata] = Field(default=None, alias="_metadata")
    tool_call_id: str  # Links back to the tool call

    @classmethod
    def create(
        cls,
        name: str,
        content: str,
        tool_call_id: str,
        action_id: str,
        error: Optional[str] = None,
    ) -> "ObservationMessage":
        """
        Helper to create an ObservationMessage with automatic metadata.

        Args:
            name: Tool/function name that was executed
            content: The execution result
            tool_call_id: ID of the tool call this responds to
            action_id: Action identifier
            error: Optional error message if execution failed

        Returns:
            ObservationMessage instance
        """
        metadata = ObservationMetadata(
            error=error,
            action_id=action_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        return cls(
            name=name,
            role="tool",
            content=content,
            metadata=metadata,
            tool_call_id=tool_call_id,
        )


# ===== Example Usage =====

if __name__ == "__main__":
    """Example usage of the message models"""

    # Example 1: Create an action message with tool calls
    action = ActionMessage.create(
        content="I'll execute a bash command to list files",
        action_id="action_exp_123_1",
        action_type="execute_bash",
        tool_calls=[
            {
                "id": "call_action_exp_123_1",
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "arguments": '{"command": "ls -la", "thought": "List files"}',
                },
            }
        ],
        completion_details={
            "id": "chatcmpl-123",
            "model": "claude-sonnet-4-5-20250929",
            "usage": {
                "total_tokens": 3269,
                "cached_tokens": 0,
                "prompt_tokens": 3137,
                "completion_tokens": 132,
            },
            "object": "chat.completion",
            "created": 1769095097,
            "completion_cost": 0.011391,
        },
    )

    print("Action Message:")
    print(action.model_dump_json(indent=2, by_alias=True))
    print()

    # Example 2: Create an observation message
    observation = ObservationMessage.create(
        name="execute_bash",
        content="Exit code: 0\nStdout:\ntotal 16\n...",
        tool_call_id="call_action_exp_123_1",
        action_id="action_exp_123_1",
        error=None,
    )

    print("Observation Message:")
    print(observation.model_dump_json(indent=2, by_alias=True))
