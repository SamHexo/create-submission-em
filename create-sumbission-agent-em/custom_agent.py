"""
Custom Agent: create-submission-agent-em

Runs every Python file in the workspace folder as a separate step.
Each script must accept --train-dataset-path and --test-dataset-path arguments
and produce a submission_<script_stem>.csv file.

After each script runs, the submission is validated against the sample submission
and graded (MAE for ventilator-pressure-prediction). Results are reported to Emily.

agent_config fields:
    competition_id       (str)  Competition folder name, e.g. "ventilator-pressure-prediction"
    train_dataset_path   (str)  Path to train CSV. Defaults to <workspace>/train.csv
    additional_args      (list) Extra CLI args appended to every script invocation
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
    """
    Return the project root:
      - /  when running inside Emily (where /workspace exists)
      - the directory containing this script when running locally
    """
    if Path("/workspace").is_dir():
        return Path("/")
    return Path(__file__).parent


def _get_workspace_dir() -> Path:
    return _get_base_dir() / "workspace"


def _get_competition_dir(competition_id: str) -> Path:
    return _get_base_dir() / competition_id


# ---------------------------------------------------------------------------
# CUDA memory
# ---------------------------------------------------------------------------

def _clear_cuda_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("  [cuda] memory cleared")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Submission grading
# ---------------------------------------------------------------------------

def _validate_and_grade(
    submission_path: Path,
    sample_submission_path: Path,
    private_test_path: Path,
    grade_py_path: Path,
) -> Tuple[bool, Optional[float], str]:
    """
    Returns (format_valid, score_or_None, human_readable_message).
    score is None when format is invalid or grading failed.
    """
    # --- 1. Load files ---
    try:
        submission = pd.read_csv(submission_path)
    except Exception as e:
        return False, None, f"Cannot read submission file: {e}"

    try:
        sample = pd.read_csv(sample_submission_path)
    except Exception as e:
        return False, None, f"Cannot read sample submission: {e}"

    # --- 2. Format checks ---
    if set(submission.columns) != set(sample.columns):
        return (
            False,
            None,
            f"Column mismatch — expected {sorted(sample.columns.tolist())}, "
            f"got {sorted(submission.columns.tolist())}",
        )

    if len(submission) != len(sample):
        return (
            False,
            None,
            f"Row count mismatch — expected {len(sample)}, got {len(submission)}",
        )

    # --- 3. Grade ---
    if not private_test_path.exists():
        return True, None, "Valid format but private_test.csv not found — skipping grade"

    if not grade_py_path.exists():
        return True, None, "Valid format but grade.py not found — skipping grade"

    try:
        private_test = pd.read_csv(private_test_path)

        spec = importlib.util.spec_from_file_location("_competition_grade", grade_py_path)
        grade_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grade_module)

        score = grade_module.grade(submission, private_test)
        return True, float(score), f"MAE = {score:.6f}"
    except Exception as e:
        return True, None, f"Valid format but grading raised an error: {e}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CreateSubmissionAgentEmAgent(BaseAgent):
    """
    For each Python file in workspace (one file per step):
      1. Runs:  python <file>
                  --train-dataset-path <train>
                  --test-dataset-path  <test>
                  --output-submission-path <workspace>/submissions/submission_<stem>.csv
                  [extra_args]
      2. Reads the submission from the explicit output path it just passed
      3. Validates & grades submission
      4. Reports score to Emily via send_iteration_result / send_experiment_completed

    Scripts must honour --output-submission-path so they can also be run
    standalone locally with a custom path.
    """

    async def start(self):
        competition_id: str = self.agent_config.get(
            "competition_id", "ventilator-pressure-prediction"
        )
        additional_args: List[str] = self.agent_config.get("additional_args", [])

        workspace_dir = _get_workspace_dir()
        competition_dir = _get_competition_dir(competition_id)

        # Paths inside competition folder
        test_dataset_path = competition_dir / "test.csv"
        sample_submission_path = competition_dir / "sample_submission.csv"
        private_test_path = competition_dir / "private_test.csv"
        grade_py_path = competition_dir / "grade.py"

        # train dataset: explicit config or default to workspace/train.csv
        train_dataset_path = Path(
            self.agent_config.get("train_dataset_path", str(workspace_dir / "train.csv"))
        )

        # Submissions land in workspace/submissions/
        submissions_dir = workspace_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)

        # Collect Python files in workspace (sorted = deterministic order)
        python_files: List[Path] = sorted(workspace_dir.glob("*.py"))

        effective_steps = min(self.max_steps, len(python_files))

        print(f"Workspace:        {workspace_dir}")
        print(f"Submissions dir:  {submissions_dir}")
        print(f"Competition dir:  {competition_dir}")
        print(f"Train dataset:    {train_dataset_path}")
        print(f"Test dataset:     {test_dataset_path}")
        print(f"Python files:     {len(python_files)}")
        print(f"Effective steps:  {effective_steps}")

        await self.send_initial_messages(
            system_message=(
                "You are an ML submission runner. "
                "Each step executes one Python training script and grades its submission."
            ),
            user_message=(
                f"Competition: {competition_id}\n"
                f"Running {effective_steps} script(s) from workspace.\n"
                f"Train data: {train_dataset_path}\n"
                f"Test data:  {test_dataset_path}"
            ),
            step_number=0,
        )

        best_score: Optional[float] = None
        best_script: Optional[str] = None

        for step, py_file in enumerate(python_files[:effective_steps], 1):
            if self.is_aborted:
                break

            print(f"\n{'='*60}")
            print(f"Step {step}/{effective_steps}: {py_file.name}")
            print(f"{'='*60}")

            # Clear GPU memory before each step
            _clear_cuda_memory()

            # Output path that the script must write to
            submission_path = submissions_dir / f"submission_{py_file.stem}.csv"

            # Build command
            cmd = [
                sys.executable,
                str(py_file),
                "--train-dataset-path", str(train_dataset_path),
                "--test-dataset-path", str(test_dataset_path),
                "--output-submission-path", str(submission_path),
            ] + [str(a) for a in additional_args]

            action_id = f"action_{self.experiment_id}_{step}"
            thought = (
                f"Running {py_file.name} with train={train_dataset_path.name}, "
                f"test={test_dataset_path.name}"
            )

            await self.send_action_received(
                step_number=step,
                action_id=action_id,
                action_type="run_script",
                action_message={
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": None,
                    "completion_details": None,
                },
            )

            # --- Execute the script ---
            observation_lines = [f"$ {' '.join(cmd)}\n"]
            score: Optional[float] = None
            error_msg: Optional[str] = None

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hours per script
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if stdout:
                    observation_lines.append(f"[stdout]\n{stdout}")
                if stderr:
                    observation_lines.append(f"[stderr]\n{stderr}")
                observation_lines.append(f"\nExit code: {proc.returncode}")

                if proc.returncode != 0:
                    error_msg = f"Script exited with code {proc.returncode}"

            except subprocess.TimeoutExpired:
                observation_lines.append("TIMEOUT: script exceeded 2-hour limit")
                error_msg = "Script timed out"
            except Exception as e:
                observation_lines.append(f"ERROR launching script: {e}")
                error_msg = str(e)

            # --- Find submission file (path was passed explicitly to the script) ---
            format_valid = False
            grade_msg = "submission file not found"

            if submission_path.exists():
                format_valid, score, grade_msg = _validate_and_grade(
                    submission_path=submission_path,
                    sample_submission_path=sample_submission_path,
                    private_test_path=private_test_path,
                    grade_py_path=grade_py_path,
                )
                observation_lines.append(f"\n[grading] {submission_path.name}: {grade_msg}")
                if not format_valid:
                    observation_lines.append("  → Submission format invalid, skipping grade")
            else:
                observation_lines.append(f"\n[grading] {submission_path.name} not found — skipping")

            observation = "\n".join(observation_lines)
            print(observation)

            # Track best (lower MAE is better)
            if score is not None:
                if best_score is None or score < best_score:
                    best_score = score
                    best_script = py_file.name

            await self.send_step_finished(
                step_number=step,
                action_id=action_id,
                action_type="run_script",
                observation_content=observation,
                tool_call_id=f"call_{step}",
                error=error_msg,
            )

            # Report iteration result (experiment stays RUNNING)
            await self.send_iteration_result(
                success=(error_msg is None and format_valid),
                summary=(
                    f"{py_file.name}: {grade_msg}"
                    if submission_path.exists()
                    else f"{py_file.name}: no submission produced"
                ),
                score=score,
                approach=f"Script: {py_file.name}",
                step_number=step,
            )

            self.current_step = step

        # Final result
        await self.send_experiment_completed(
            success=best_score is not None,
            summary=(
                f"Best score: {best_score:.6f} (MAE) from {best_script}"
                if best_score is not None
                else "No valid graded submission produced"
            ),
            score=best_score,
            approach=f"Ran {effective_steps} script(s) from workspace",
            step_number=self.current_step,
        )

        print(f"\nDone. Best MAE: {best_score} ({best_script})")

    async def continue_agent(
        self,
        user_message: str,
        new_max_steps: int,
        step_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ):
        print(f"Received user message: {user_message}")
        self.max_steps = new_max_steps

        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1

        await self.send_step_finished(
            step_number=self.current_step,
            user_message=user_message,
            git_branch=branch_name,
        )

        # Re-run the remaining scripts with the new max_steps budget
        await self.start()

    async def abort(self):
        print("Aborting create-submission-agent-em...")
        self.is_aborted = True
        await self.send_experiment_aborted(
            reason="Aborted by user",
            last_step=self.current_step,
        )


def create_agent(
    experiment_id: str,
    project_id: str,
    problem_statement: str,
    max_steps: int,
    api_keys: Dict[str, str],
    webhook_url: Optional[str] = None,
    agent_config: Dict[str, Any] = None,
    jwt_token: str = None,
) -> BaseAgent:
    return CreateSubmissionAgentEmAgent(
        experiment_id=experiment_id,
        project_id=project_id,
        problem_statement=problem_statement,
        max_steps=max_steps,
        api_keys=api_keys,
        webhook_url=webhook_url,
        agent_config=agent_config,
        jwt_token=jwt_token,
    )
