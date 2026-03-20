"""
Custom Agent: create-submission-agent-em

Iterates over Python files in <code_folder> (one file = one step), sorted
alphabetically. Runs at most max_steps scripts; stops when all files are done.

Each script must accept:
    --train-dataset-path        path to train CSV
    --test-dataset-path         path to competition test CSV
    --output-submission-path    where to write the submission CSV
    (+ any additional_args)

After each script the agent validates and grades the submission, then reports
the MAE score to Emily via send_iteration_result.

agent_config fields:
    competition_id       (str)   Competition folder name.
                                 Default: "ventilator-pressure-prediction"
    code_folder_path     (str)   Path to the folder containing scripts to run.
                                 Absolute, or relative to the agent script.
                                 Default: <workspace>/codes
    train_dataset_path   (str)   Path to train CSV.
                                 Default: <workspace>/train.csv
    additional_args      (list)  Extra CLI args appended to every invocation.
                                 Default: []
    parallelism          (int)   Number of scripts to run simultaneously.
                                 Scripts are batched in groups of this size;
                                 within a batch all run concurrently and results
                                 are reported as each one finishes.
                                 Default: 1 (sequential)
    timeout_per_script   (int)   Hard timeout in seconds per script.
                                 The subprocess is killed if it exceeds this.
                                 Default: 7200 (2 hours)
    max_ram_gb           (float) Kill a script if its RAM usage (RSS, including
                                 child processes) exceeds this many GB.
                                 Default: None (no limit)
    ram_check_interval   (float) How often (seconds) to poll RAM usage.
                                 Default: 5.0
    only_files           (list)  Optional list of filenames (e.g. ["family1_deep_res_bilstm.py"])
                                 to run. If set, only these files are executed (in the given order);
                                 all other .py files in the code folder are ignored.
                                 Default: None (run all files)
    checkpoint_steps     (list)  List of gradient step values for FULL checkpoints:
                                 submission CSV + grade + Emily step.
                                 Default: None
    patience_every       (int)   Interval (gradient steps) for lightweight patience checks:
                                 val score only, no submission, no grade, no Emily step.
                                 Goes up to the last value in checkpoint_steps, or up to
                                 --gradient-steps from additional_args if checkpoint_steps
                                 is null (in which case the final step is a full checkpoint).
                                 Default: None
    early_stopping_patience (int) Stop a script early if val score has not improved for
                                 this many consecutive patience checks (or full checkpoints
                                 if patience_every is not set).
                                 Default: None (disabled)
    checkpoint_order     (str)   Controls the order in which checkpoint steps and scripts
                                 are executed. Only applies when checkpoint_steps is set.
                                 "by_script" (default): outer loop = scripts, inner = steps.
                                   Each script runs all its checkpoint steps before the next
                                   script starts. Natural with parallelism > 1.
                                 "by_step": outer loop = steps, inner = scripts.
                                   All scripts run at step N before any script advances to
                                   step N+1. Only sequential (parallelism=1). Useful when
                                   you want to compare all scripts at the same step before
                                   committing to the next level.
                                 Default: "by_script"
"""

import asyncio
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re as _re

import pandas as pd

from base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Val score parser
# ---------------------------------------------------------------------------

def _parse_val_score(stdout: Optional[str]) -> Optional[float]:
    """Parse 'Final Validation Score: X.XXXX' from script stdout."""
    if not stdout:
        return None
    m = _re.search(r"Final Validation Score:\s*([\d.eE+\-]+)", stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_gradient_steps(additional_args: List[str]) -> Optional[int]:
    """Extract --gradient-steps value from additional_args list."""
    for i, a in enumerate(additional_args):
        if a == "--gradient-steps" and i + 1 < len(additional_args):
            try:
                return int(additional_args[i + 1])
            except ValueError:
                pass
    return None


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
    return Path(__file__).parent / competition_id


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_archives(code_folder: Path) -> None:
    """
    Extract any zip / tar / tar.gz / tgz / tar.bz2 / gz archives found
    directly in code_folder, then flatten all resulting .py files to the
    root of code_folder (overwrite on name collision).
    Archives and temporary sub-directories are removed after extraction.
    """
    import gzip
    import shutil
    import tarfile
    import zipfile

    ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2"}

    archives = [
        f for f in code_folder.iterdir()
        if f.is_file() and f.suffix.lower() in ARCHIVE_SUFFIXES
    ]

    if not archives:
        return

    for archive in archives:
        tmp_dir = code_folder / f"_tmp_{archive.stem}"
        tmp_dir.mkdir()
        name = archive.name.lower()

        try:
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(tmp_dir)

            elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
                with tarfile.open(archive) as tf:
                    tf.extractall(tmp_dir)

            elif name.endswith(".gz"):
                # Single-file gzip (e.g. script.py.gz)
                out_name = archive.stem  # strips the .gz
                with gzip.open(archive, "rb") as gz_in:
                    (tmp_dir / out_name).write_bytes(gz_in.read())

            print(f"  [extract] {archive.name} → {tmp_dir.name}/")

            # Flatten: move every .py file from the extracted tree to code_folder
            for py_file in tmp_dir.rglob("*.py"):
                dest = code_folder / py_file.name
                shutil.move(str(py_file), str(dest))
                print(f"    ← {py_file.relative_to(tmp_dir)}  →  {dest.name}")

            # Clean up macOS AppleDouble metadata files (._foo.py) injected by
            # Finder/Archive Utility. If the real foo.py also exists → delete ._foo.py.
            # If only ._foo.py exists (edge case) → rename it to foo.py.
            for dot_file in list(code_folder.glob("._*.py")):
                real_name = dot_file.name[2:]  # strip leading ._
                real_file = code_folder / real_name
                if real_file.exists():
                    dot_file.unlink()
                    print(f"    [cleanup] removed macOS metadata file: {dot_file.name}")
                else:
                    dot_file.rename(real_file)
                    print(f"    [cleanup] renamed {dot_file.name} → {real_name}")

        except Exception as e:
            print(f"  [extract] WARNING: could not extract {archive.name}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            archive.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


# ---------------------------------------------------------------------------
# Async script runner
# ---------------------------------------------------------------------------

async def _run_script_async(
    cmd: List[str],
    timeout: Optional[int] = 7200,
    env: Optional[Dict[str, str]] = None,
    max_ram_bytes: Optional[int] = None,
    ram_check_interval: float = 5.0,
) -> Tuple[Optional[str], Optional[str], int, float, Optional[str]]:
    """
    Run a script asynchronously.
    Returns (stdout, stderr, returncode, elapsed_seconds, error_msg).
    error_msg is None on success.

    - env: overrides for environment variables (e.g. CUDA_VISIBLE_DEVICES)
    - timeout: hard wall-clock limit in seconds; process is killed if exceeded
    - max_ram_bytes: kill the process if its RSS (incl. children) exceeds this
    - ram_check_interval: how often (seconds) to poll RAM

    Each subprocess owns its CUDA context; all VRAM is freed by the OS on exit.
    """
    t0 = asyncio.get_event_loop().time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as e:
        elapsed = round(asyncio.get_event_loop().time() - t0, 2)
        return None, None, -1, elapsed, str(e)

    communicate_task = asyncio.ensure_future(proc.communicate())

    # Optional RAM monitor: polls RSS of the subprocess and its children
    monitor_task: Optional[asyncio.Task] = None
    if max_ram_bytes is not None:
        async def _ram_monitor() -> Optional[int]:
            try:
                import psutil
                ps = psutil.Process(proc.pid)
                while True:
                    await asyncio.sleep(ram_check_interval)
                    try:
                        rss = ps.memory_info().rss
                        for child in ps.children(recursive=True):
                            try:
                                rss += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        if rss > max_ram_bytes:
                            return rss
                    except psutil.NoSuchProcess:
                        return None  # process already gone
            except Exception:
                return None

        monitor_task = asyncio.ensure_future(_ram_monitor())

    wait_tasks = [communicate_task] + ([monitor_task] if monitor_task else [])

    done, pending = await asyncio.wait(
        wait_tasks,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    elapsed = round(asyncio.get_event_loop().time() - t0, 2)

    async def _drain_and_cancel():
        """Kill proc, drain pipes, cancel remaining tasks."""
        proc.kill()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(communicate_task, timeout=10.0)
        except Exception:
            stdout_b, stderr_b = b"", b""
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        return stdout_b, stderr_b

    # ── Timeout ────────────────────────────────────────────────────────────
    if not done:
        stdout_b, stderr_b = await _drain_and_cancel()
        return (
            stdout_b.decode(errors="replace") or None,
            stderr_b.decode(errors="replace") or None,
            -1, elapsed,
            f"Killed: timeout after {timeout}s",
        )

    # ── RAM limit exceeded ──────────────────────────────────────────────────
    if monitor_task is not None and monitor_task in done:
        rss = monitor_task.result()
        stdout_b, stderr_b = await _drain_and_cancel()
        rss_gb = (rss or 0) / 1024 ** 3
        limit_gb = max_ram_bytes / 1024 ** 3
        return (
            stdout_b.decode(errors="replace") or None,
            stderr_b.decode(errors="replace") or None,
            -1, elapsed,
            f"Killed: RAM {rss_gb:.1f}GB exceeded limit {limit_gb:.1f}GB",
        )

    # ── Normal completion ───────────────────────────────────────────────────
    # Cancel the RAM monitor if it's still sleeping
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    stdout_bytes, stderr_bytes = communicate_task.result()
    return (
        stdout_bytes.decode(errors="replace"),
        stderr_bytes.decode(errors="replace"),
        proc.returncode,
        elapsed,
        None,
    )


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
    For each Python file in <code_folder> (alphabetical, one file per step):
      1. Runs:  python <file>
                  --train-dataset-path      <train>
                  --test-dataset-path       <test>
                  --output-submission-path  <workspace>/submissions/submission_<stem>.csv
                  [additional_args]
      2. Reads the submission from the explicit output path it just passed
      3. Validates & grades submission
      4. Reports score to Emily via send_iteration_result / send_experiment_completed

    Stops after min(max_steps, len(scripts)) steps — never loops beyond available files.
    Scripts must honour --output-submission-path so they can also be run standalone.
    """

    async def start(self):
        competition_id: str = self.agent_config.get(
            "competition_id", "ventilator-pressure-prediction"
        )
        _additional_args_raw = self.agent_config.get("additional_args", [])
        additional_args: List[str] = _additional_args_raw if _additional_args_raw else []
        _parallelism_raw = self.agent_config.get("parallelism", 1)
        parallelism: int = max(1, int(_parallelism_raw)) if _parallelism_raw else 1
        _timeout_raw = self.agent_config.get("timeout_per_script", 7200)
        timeout_per_script: Optional[int] = int(_timeout_raw) if _timeout_raw else None
        _max_ram_raw = self.agent_config.get("max_ram_gb")
        max_ram_gb: Optional[float] = float(_max_ram_raw) if _max_ram_raw else None
        _ram_interval_raw = self.agent_config.get("ram_check_interval", 5.0)
        ram_check_interval: float = float(_ram_interval_raw) if _ram_interval_raw else 0.0
        # ram_check_interval: 0 ou max_ram_gb: 0/null → monitoring désactivé
        if not ram_check_interval:
            max_ram_gb = None
        max_ram_bytes: Optional[int] = int(max_ram_gb * 1024 ** 3) if max_ram_gb else None

        # Checkpoint config
        _ckpt_steps_raw = self.agent_config.get("checkpoint_steps")
        _patience_every = self.agent_config.get("patience_every")

        # Normalise: [], 0, "" → None
        if isinstance(_ckpt_steps_raw, list) and len(_ckpt_steps_raw) == 0:
            _ckpt_steps_raw = None
        if _ckpt_steps_raw:
            if isinstance(_ckpt_steps_raw, (int, float)):
                checkpoint_steps_set = {int(_ckpt_steps_raw)} if int(_ckpt_steps_raw) > 0 else None
            else:
                checkpoint_steps_set = {int(s) for s in _ckpt_steps_raw if int(s) > 0} or None
        else:
            checkpoint_steps_set = None

        _early_stopping_raw = self.agent_config.get("early_stopping_patience")
        early_stopping_patience: Optional[int] = int(_early_stopping_raw) if _early_stopping_raw else None
        if early_stopping_patience == 0:
            early_stopping_patience = None

        _patience_every_int: Optional[int] = int(_patience_every) if _patience_every else None
        if _patience_every_int == 0:
            _patience_every_int = None

        _ckpt_order_raw = self.agent_config.get("checkpoint_order", "by_script")
        checkpoint_order = _ckpt_order_raw if _ckpt_order_raw in ("by_script", "by_step") else "by_script"

        workspace_dir = _get_workspace_dir()
        competition_dir = _get_competition_dir(competition_id)

        # Code folder: explicit config, or default to workspace/codes
        raw_code_folder = self.agent_config.get("code_folder_path")
        if raw_code_folder:
            code_folder = Path(raw_code_folder)
            if not code_folder.is_absolute():
                # Relative paths are anchored to the project root (same as workspace)
                code_folder = _get_base_dir() / code_folder
        else:
            code_folder = workspace_dir / "codes"

        # Paths inside competition folder
        test_dataset_path = competition_dir / "test.csv"
        sample_submission_path = competition_dir / "sample_submission.csv"
        private_test_path = competition_dir / "private_test.csv"
        grade_py_path = competition_dir / "grade.py"

        # train dataset: explicit config or default to workspace/train.csv
        raw_train_path = self.agent_config.get("train_dataset_path")
        if raw_train_path:
            train_dataset_path = Path(raw_train_path)
            if not train_dataset_path.is_absolute():
                train_dataset_path = _get_base_dir() / train_dataset_path
        else:
            train_dataset_path = workspace_dir / "train.csv"

        def _make_run_dir(base_name: str) -> Path:
            candidate = workspace_dir / base_name
            if candidate.exists():
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                candidate = workspace_dir / f"{base_name}_{ts}"
            candidate.mkdir(parents=True)
            return candidate

        submissions_dir = _make_run_dir(f"submissions_{self.experiment_id}")
        grades_dir = _make_run_dir(f"grades_{self.experiment_id}")

        _only_files_raw = self.agent_config.get("only_files")
        # Accept both a single string and a list (YAML scalar vs sequence)
        if isinstance(_only_files_raw, str):
            only_files: Optional[List[str]] = [_only_files_raw] if _only_files_raw else None
        elif isinstance(_only_files_raw, list):
            only_files = _only_files_raw if _only_files_raw else None
        else:
            only_files = None

        # Extract any archives in code folder before iterating
        _extract_archives(code_folder)

        # Collect Python files in code folder (alphabetical = deterministic order)
        available = [f.name for f in code_folder.glob("*.py") if f.is_file()]
        print(f"Files in code folder after extraction: {sorted(available)}")
        if only_files:
            print(f"only_files filter: {only_files}")
            python_files = []
            for f in only_files:
                p = code_folder / f
                if p.is_file():
                    python_files.append(p)
                else:
                    print(f"  WARNING: only_files entry '{f}' not found in {code_folder} — skipping")
            print(f"Matched files: {[f.name for f in python_files]}")
        else:
            python_files = sorted(p for p in code_folder.glob("*.py") if p.is_file())


        # Run at most max_steps scripts; stop when files are exhausted
        effective_steps = min(self.max_steps, len(python_files))

        num_gpus = _count_gpus()

        print(f"Workspace:        {workspace_dir}")
        print(f"Code folder:      {code_folder}  ({len(python_files)} script(s))")
        print(f"Submissions dir:  {submissions_dir}")
        print(f"Competition dir:  {competition_dir}")
        print(f"Train dataset:    {train_dataset_path}")
        print(f"Test dataset:     {test_dataset_path}")
        print(f"Python files:     {len(python_files)}")
        print(f"Effective steps:  {effective_steps}")
        print(f"Parallelism:      {parallelism}")
        print(f"GPUs available:   {num_gpus}")

        await self.send_initial_messages(
            system_message=(
                "You are an ML submission runner. "
                "Each step executes one Python training script and grades its submission."
            ),
            user_message=(
                f"Competition: {competition_id}\n"
                f"Running {effective_steps}/{len(python_files)} script(s) from {code_folder.name}.\n"
                f"Train data: {train_dataset_path}\n"
                f"Test data:  {test_dataset_path}\n"
                f"Parallelism: {parallelism}"
            ),
            step_number=0,
        )

        best_score: Optional[float] = None
        best_script: Optional[str] = None

        # Split scripts into batches of `parallelism` size
        scripts_to_run = python_files[:effective_steps]
        batches = [
            scripts_to_run[i : i + parallelism]
            for i in range(0, len(scripts_to_run), parallelism)
        ]

        global_step = 0  # tracks step index across all batches

        import os as _os
        base_env = _os.environ.copy()

        if checkpoint_steps_set is not None or _patience_every_int is not None:
            # Build per-script timeline of (gradient_step, is_full_checkpoint)
            # is_full_checkpoint=True → submission CSV + grade + Emily step
            # is_full_checkpoint=False → val score only (patience check), no submission/grade/step

            # Filter --gradient-steps from additional_args (we manage it ourselves)
            filtered_additional_args = []
            skip_next = False
            for a in [str(x) for x in additional_args]:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--gradient-steps":
                    skip_next = True
                    continue
                filtered_additional_args.append(a)

            checkpoint_dir = submissions_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            for py_file in (scripts_to_run if checkpoint_order != "by_step" else []):
                if self.is_aborted:
                    break

                # Resolve full checkpoint steps for this script
                if checkpoint_steps_set:
                    full_steps = sorted(checkpoint_steps_set)
                else:
                    # No checkpoint_steps: use --gradient-steps from additional_args as the single full checkpoint
                    gs = _extract_gradient_steps([str(a) for a in additional_args])
                    full_steps = [gs] if gs else []

                if not full_steps:
                    print(f"WARNING: No gradient steps resolved for {py_file.name}, skipping")
                    continue

                max_step = max(full_steps)
                full_steps_set = set(full_steps)

                # Build merged timeline
                if _patience_every_int:
                    all_steps = sorted(set(
                        list(range(_patience_every_int, max_step, _patience_every_int)) +
                        list(full_steps_set) +
                        [max_step]
                    ))
                else:
                    all_steps = sorted(full_steps_set)

                timeline = [(s, s in full_steps_set) for s in all_steps]

                checkpoint_path = checkpoint_dir / f"{py_file.stem}.ckpt"
                best_ckpt_val: Optional[float] = None
                patience_counter = 0
                early_stopped = False

                for ckpt_step, is_full in timeline:
                    if self.is_aborted or early_stopped:
                        break

                    cmd = [
                        sys.executable,
                        str(py_file),
                        "--train-dataset-path", str(train_dataset_path),
                        "--test-dataset-path", str(test_dataset_path),
                        "--gradient-steps", str(ckpt_step),
                        "--checkpoint-path", str(checkpoint_path),
                    ] + filtered_additional_args

                    if is_full:
                        global_step += 1
                        step = global_step
                        submission_path = submissions_dir / f"submission_{py_file.stem}_step{ckpt_step}.csv"
                        action_id = f"action_{self.experiment_id}_{step}"
                        cmd += ["--output-submission-path", str(submission_path)]

                        thought = (
                            f"Running {py_file.name} — full checkpoint at step {ckpt_step} "
                            f"(train={train_dataset_path.name}, test={test_dataset_path.name})"
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
                    else:
                        submission_path = None

                    check_type = "FULL" if is_full else "patience"
                    print(f"\n{'='*60}")
                    print(f"{check_type}: {py_file.name} @ gradient_steps={ckpt_step}")
                    print(f"{'='*60}")

                    stdout, stderr, returncode, elapsed_seconds, run_error = await _run_script_async(
                        cmd,
                        timeout=timeout_per_script,
                        env=base_env,
                        max_ram_bytes=max_ram_bytes,
                        ram_check_interval=ram_check_interval,
                    )

                    # Early stopping: parse val score from stdout
                    val_score = _parse_val_score(stdout)
                    check_type_label = "FULL checkpoint" if is_full else "patience check"

                    # Always log val score
                    if val_score is not None:
                        improved = best_ckpt_val is None or val_score < best_ckpt_val
                        if improved:
                            delta_str = f" (↓ {best_ckpt_val - val_score:.4f})" if best_ckpt_val is not None else " (first)"
                            best_ckpt_val = val_score
                            patience_counter = 0
                            print(f"  [{check_type_label}] step={ckpt_step}, val={val_score:.4f}{delta_str} ✓ new best, elapsed={elapsed_seconds}s")
                        else:
                            patience_counter += 1
                            patience_str = f"  patience={patience_counter}/{early_stopping_patience}" if early_stopping_patience else ""
                            print(f"  [{check_type_label}] step={ckpt_step}, val={val_score:.4f} (best={best_ckpt_val:.4f}){patience_str}, elapsed={elapsed_seconds}s")
                            if early_stopping_patience and patience_counter >= early_stopping_patience:
                                print(f"  [early stopping] No improvement for {patience_counter} checks. Stopping {py_file.name}.")
                                early_stopped = True
                    else:
                        print(f"  [{check_type_label}] step={ckpt_step}, val=N/A (could not parse), elapsed={elapsed_seconds}s")
                        if run_error:
                            print(f"  ERROR: {run_error}")

                    if not is_full:
                        # Patience-only: log and continue, no Emily step
                        continue

                    # Full checkpoint: grade + Emily step
                    observation_lines = [f"$ {' '.join(cmd)}\n"]
                    error_msg: Optional[str] = run_error
                    score: Optional[float] = None

                    if run_error is None:
                        if stdout and stdout.strip():
                            stdout_lines = stdout.strip().splitlines()
                            if len(stdout_lines) > 30:
                                stdout_tail = "\n".join(stdout_lines[-30:])
                                observation_lines.append(f"[stdout] (last 30 of {len(stdout_lines)} lines)\n{stdout_tail}")
                            else:
                                observation_lines.append(f"[stdout]\n{stdout.strip()}")
                        if stderr and stderr.strip():
                            observation_lines.append(f"[stderr]\n{stderr.strip()}")
                        observation_lines.append(f"\nExit code: {returncode}  ({elapsed_seconds}s)")
                        if returncode != 0:
                            error_msg = f"Script exited with code {returncode}"
                    else:
                        observation_lines.append(f"ERROR: {run_error}  ({elapsed_seconds}s)")

                    format_valid = False
                    grade_msg = "submission file not found"

                    if submission_path and submission_path.exists():
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
                        observation_lines.append(f"\n[grading] submission not found — skipping")

                    if early_stopped:
                        observation_lines.append(
                            f"\n[early stopping] Stopped after {patience_counter} checks without improvement "
                            f"(best val={best_ckpt_val:.4f})."
                        )

                    observation = "\n".join(observation_lines)
                    print(f"\n[step {step}] {py_file.name} @{ckpt_step} finished ({elapsed_seconds}s)")
                    print(observation)

                    grades_entry = {
                        "python_file": str(py_file),
                        "gradient_steps": ckpt_step,
                        "submission_file": str(submission_path) if submission_path and submission_path.exists() else None,
                        "score": score,
                        "format_valid": format_valid,
                        "grade_message": grade_msg,
                        "execution_time_seconds": elapsed_seconds,
                        "error": error_msg,
                        "early_stopped": early_stopped,
                    }
                    grades_path = grades_dir / f"metric_{py_file.stem}_step{ckpt_step}.json"
                    grades_path.write_text(json.dumps(grades_entry, indent=2))
                    print(f"  [grades] {grades_path.name} written")

                    if score is not None:
                        if best_score is None or score < best_score:
                            best_score = score
                            best_script = f"{py_file.name}@step{ckpt_step}"

                    await self.send_step_finished(
                        step_number=step,
                        action_id=action_id,
                        action_type="run_script",
                        observation_content=observation,
                        tool_call_id=f"call_{step}",
                        error=error_msg,
                    )

                    await self.send_iteration_result(
                        success=(error_msg is None and format_valid),
                        summary=(
                            f"{py_file.name}@step{ckpt_step}: {grade_msg}"
                            if submission_path and submission_path.exists()
                            else f"{py_file.name}@step{ckpt_step}: no submission produced"
                        ),
                        score=score,
                        approach=f"Script: {py_file.name}, gradient_steps={ckpt_step}",
                        step_number=step,
                    )

                    self.current_step = max(self.current_step, step)

            if checkpoint_order == "by_step":
                # Build global timeline (shared across all scripts — same checkpoint_steps for all)
                if checkpoint_steps_set:
                    _bs_full_steps = sorted(checkpoint_steps_set)
                else:
                    _bs_gs = _extract_gradient_steps([str(a) for a in additional_args])
                    _bs_full_steps = [_bs_gs] if _bs_gs else []

                if not _bs_full_steps:
                    print("WARNING: [by_step] No gradient steps resolved, skipping")
                else:
                    _bs_max = max(_bs_full_steps)
                    _bs_full_set = set(_bs_full_steps)
                    if _patience_every_int:
                        _bs_all = sorted(set(
                            list(range(_patience_every_int, _bs_max, _patience_every_int)) +
                            list(_bs_full_set) + [_bs_max]
                        ))
                    else:
                        _bs_all = sorted(_bs_full_set)
                    _bs_timeline = [(_s, _s in _bs_full_set) for _s in _bs_all]

                    # Per-script state
                    _bs_ckpt_path = {pf: checkpoint_dir / f"{pf.stem}.ckpt" for pf in scripts_to_run}
                    _bs_best_val: Dict[Path, Optional[float]] = {pf: None for pf in scripts_to_run}
                    _bs_patience: Dict[Path, int] = {pf: 0 for pf in scripts_to_run}
                    _bs_stopped: Dict[Path, bool] = {pf: False for pf in scripts_to_run}

                    for ckpt_step, is_full in _bs_timeline:
                        if self.is_aborted:
                            break
                        for py_file in scripts_to_run:
                            if self.is_aborted or _bs_stopped[py_file]:
                                continue

                            checkpoint_path = _bs_ckpt_path[py_file]
                            best_ckpt_val = _bs_best_val[py_file]
                            patience_counter = _bs_patience[py_file]
                            early_stopped = False

                            cmd = [
                                sys.executable,
                                str(py_file),
                                "--train-dataset-path", str(train_dataset_path),
                                "--test-dataset-path", str(test_dataset_path),
                                "--gradient-steps", str(ckpt_step),
                                "--checkpoint-path", str(checkpoint_path),
                            ] + filtered_additional_args

                            if is_full:
                                global_step += 1
                                step = global_step
                                submission_path = submissions_dir / f"submission_{py_file.stem}_step{ckpt_step}.csv"
                                action_id = f"action_{self.experiment_id}_{step}"
                                cmd += ["--output-submission-path", str(submission_path)]

                                thought = (
                                    f"Running {py_file.name} — full checkpoint at step {ckpt_step} "
                                    f"(train={train_dataset_path.name}, test={test_dataset_path.name})"
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
                            else:
                                submission_path = None

                            check_type = "FULL" if is_full else "patience"
                            print(f"\n{'='*60}")
                            print(f"{check_type}: {py_file.name} @ gradient_steps={ckpt_step}")
                            print(f"{'='*60}")

                            stdout, stderr, returncode, elapsed_seconds, run_error = await _run_script_async(
                                cmd,
                                timeout=timeout_per_script,
                                env=base_env,
                                max_ram_bytes=max_ram_bytes,
                                ram_check_interval=ram_check_interval,
                            )

                            val_score = _parse_val_score(stdout)
                            check_type_label = "FULL checkpoint" if is_full else "patience check"

                            if val_score is not None:
                                improved = best_ckpt_val is None or val_score < best_ckpt_val
                                if improved:
                                    delta_str = f" (↓ {best_ckpt_val - val_score:.4f})" if best_ckpt_val is not None else " (first)"
                                    best_ckpt_val = val_score
                                    patience_counter = 0
                                    print(f"  [{check_type_label}] step={ckpt_step}, val={val_score:.4f}{delta_str} ✓ new best, elapsed={elapsed_seconds}s")
                                else:
                                    patience_counter += 1
                                    patience_str = f"  patience={patience_counter}/{early_stopping_patience}" if early_stopping_patience else ""
                                    print(f"  [{check_type_label}] step={ckpt_step}, val={val_score:.4f} (best={best_ckpt_val:.4f}){patience_str}, elapsed={elapsed_seconds}s")
                                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                                        print(f"  [early stopping] No improvement for {patience_counter} checks. Stopping {py_file.name}.")
                                        early_stopped = True
                            else:
                                print(f"  [{check_type_label}] step={ckpt_step}, val=N/A (could not parse), elapsed={elapsed_seconds}s")
                                if run_error:
                                    print(f"  ERROR: {run_error}")

                            _bs_best_val[py_file] = best_ckpt_val
                            _bs_patience[py_file] = patience_counter
                            if early_stopped:
                                _bs_stopped[py_file] = True

                            if not is_full:
                                continue

                            # Full checkpoint: grade + Emily step
                            observation_lines = [f"$ {' '.join(cmd)}\n"]
                            error_msg: Optional[str] = run_error
                            score: Optional[float] = None

                            if run_error is None:
                                if stdout and stdout.strip():
                                    stdout_lines = stdout.strip().splitlines()
                                    if len(stdout_lines) > 30:
                                        stdout_tail = "\n".join(stdout_lines[-30:])
                                        observation_lines.append(f"[stdout] (last 30 of {len(stdout_lines)} lines)\n{stdout_tail}")
                                    else:
                                        observation_lines.append(f"[stdout]\n{stdout.strip()}")
                                if stderr and stderr.strip():
                                    observation_lines.append(f"[stderr]\n{stderr.strip()}")
                                observation_lines.append(f"\nExit code: {returncode}  ({elapsed_seconds}s)")
                                if returncode != 0:
                                    error_msg = f"Script exited with code {returncode}"
                            else:
                                observation_lines.append(f"ERROR: {run_error}  ({elapsed_seconds}s)")

                            format_valid = False
                            grade_msg = "submission file not found"

                            if submission_path and submission_path.exists():
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
                                observation_lines.append(f"\n[grading] submission not found — skipping")

                            if early_stopped:
                                observation_lines.append(
                                    f"\n[early stopping] Stopped after {patience_counter} checks without improvement "
                                    f"(best val={best_ckpt_val:.4f})."
                                )

                            observation = "\n".join(observation_lines)
                            print(f"\n[step {step}] {py_file.name} @{ckpt_step} finished ({elapsed_seconds}s)")
                            print(observation)

                            grades_entry = {
                                "python_file": str(py_file),
                                "gradient_steps": ckpt_step,
                                "submission_file": str(submission_path) if submission_path and submission_path.exists() else None,
                                "score": score,
                                "format_valid": format_valid,
                                "grade_message": grade_msg,
                                "execution_time_seconds": elapsed_seconds,
                                "error": error_msg,
                                "early_stopped": early_stopped,
                            }
                            grades_path = grades_dir / f"metric_{py_file.stem}_step{ckpt_step}.json"
                            grades_path.write_text(json.dumps(grades_entry, indent=2))
                            print(f"  [grades] {grades_path.name} written")

                            if score is not None:
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_script = f"{py_file.name}@step{ckpt_step}"

                            await self.send_step_finished(
                                step_number=step,
                                action_id=action_id,
                                action_type="run_script",
                                observation_content=observation,
                                tool_call_id=f"call_{step}",
                                error=error_msg,
                            )

                            await self.send_iteration_result(
                                success=(error_msg is None and format_valid),
                                summary=(
                                    f"{py_file.name}@step{ckpt_step}: {grade_msg}"
                                    if submission_path and submission_path.exists()
                                    else f"{py_file.name}@step{ckpt_step}: no submission produced"
                                ),
                                score=score,
                                approach=f"Script: {py_file.name}, gradient_steps={ckpt_step}",
                                step_number=step,
                            )

                            self.current_step = max(self.current_step, step)

        else:
            # Original batch mode (no checkpointing)
            for batch in batches:
                if self.is_aborted:
                    break

                batch_items = []
                for slot_idx, py_file in enumerate(batch):
                    global_step += 1
                    step = global_step
                    submission_path = submissions_dir / f"submission_{py_file.stem}.csv"
                    cmd = [
                        sys.executable,
                        str(py_file),
                        "--train-dataset-path", str(train_dataset_path),
                        "--test-dataset-path", str(test_dataset_path),
                        "--output-submission-path", str(submission_path),
                    ] + [str(a) for a in additional_args]
                    action_id = f"action_{self.experiment_id}_{step}"

                    # Assign a dedicated GPU to this slot when possible
                    proc_env = base_env.copy()
                    if parallelism > 1 and num_gpus > 0:
                        gpu_id = slot_idx % num_gpus
                        proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        gpu_label = f"GPU {gpu_id}"
                    else:
                        gpu_label = f"all {num_gpus} GPU(s)" if num_gpus > 0 else "CPU"

                    batch_items.append((step, py_file, cmd, action_id, submission_path, proc_env, gpu_label))

                batch_label = " + ".join(
                    f"step {s} ({f.name} → {gl})" for s, f, _c, _a, _sp, _e, gl in batch_items
                )
                print(f"\n{'='*60}")
                print(f"Batch: {batch_label}")
                print(f"{'='*60}")

                # Phase 1: send ACTION_RECEIVED for all scripts in the batch
                for step, py_file, cmd, action_id, submission_path, proc_env, gpu_label in batch_items:
                    thought = (
                        f"Running {py_file.name} on {gpu_label} — "
                        f"train={train_dataset_path.name}, test={test_dataset_path.name}"
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

                # Phase 2: launch all scripts concurrently; process results as they arrive.
                # Each subprocess has its own CUDA context (CUDA_VISIBLE_DEVICES already set
                # in proc_env). Its VRAM is freed automatically when the process exits.
                async def _run_one(
                    step: int,
                    py_file: Path,
                    cmd: List[str],
                    action_id: str,
                    submission_path: Path,
                    proc_env: Dict[str, str],
                    gpu_label: str,
                ):
                    stdout, stderr, returncode, elapsed, run_error = await _run_script_async(
                        cmd,
                        timeout=timeout_per_script,
                        env=proc_env,
                        max_ram_bytes=max_ram_bytes,
                        ram_check_interval=ram_check_interval,
                    )
                    return step, py_file, cmd, action_id, submission_path, stdout, stderr, returncode, elapsed, run_error

                tasks = [
                    asyncio.ensure_future(_run_one(*item))
                    for item in batch_items
                ]

                for coro in asyncio.as_completed(tasks):
                    (
                        step, py_file, cmd, action_id, submission_path,
                        stdout, stderr, returncode, elapsed_seconds, run_error,
                    ) = await coro

                    # Build observation
                    observation_lines = [f"$ {' '.join(cmd)}\n"]
                    error_msg: Optional[str] = run_error
                    score: Optional[float] = None

                    if run_error is None:
                        if stdout and stdout.strip():
                            stdout_lines = stdout.strip().splitlines()
                            if len(stdout_lines) > 30:
                                stdout_tail = "\n".join(stdout_lines[-30:])
                                observation_lines.append(f"[stdout] (last 30 of {len(stdout_lines)} lines)\n{stdout_tail}")
                            else:
                                observation_lines.append(f"[stdout]\n{stdout.strip()}")
                        if stderr and stderr.strip():
                            observation_lines.append(f"[stderr]\n{stderr.strip()}")
                        observation_lines.append(f"\nExit code: {returncode}  ({elapsed_seconds}s)")
                        if returncode != 0:
                            error_msg = f"Script exited with code {returncode}"
                    else:
                        observation_lines.append(f"ERROR: {run_error}  ({elapsed_seconds}s)")

                    # Grade
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
                    print(f"\n[step {step}] {py_file.name} finished ({elapsed_seconds}s)")
                    print(observation)

                    # Write grades JSON
                    grades_entry = {
                        "python_file": str(py_file),
                        "submission_file": str(submission_path) if submission_path.exists() else None,
                        "score": score,
                        "format_valid": format_valid,
                        "grade_message": grade_msg,
                        "execution_time_seconds": elapsed_seconds,
                        "error": error_msg,
                    }
                    grades_path = grades_dir / f"metric_{py_file.stem}.json"
                    grades_path.write_text(json.dumps(grades_entry, indent=2))
                    print(f"  [grades] {grades_path.name} written")

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

                    self.current_step = max(self.current_step, step)

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
