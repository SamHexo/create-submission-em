# FORMAT_CODE_PROMPT

Instructions for an AI agent to reformat a Python ML script so it can be invoked correctly from the command line — **without changing the business logic**.

---

## Objective

The script must be invocable as follows :

```bash
python script.py \
  --train-dataset-path /path/to/train.csv \
  --test-dataset-path  /path/to/test.csv \
  --output-submission-path /path/to/submission.csv \
  [--epochs 10] \
  [other optional args...]
```

---

## Instructions for the agent

You will reformat the provided Python code. **You do not change the logic** (no change to algorithm, feature engineering, default hyperparameters, or model structure). Only the CLI wiring and path handling change.

### Mandatory rules

1. **Add `argparse`** at the top of the file (if missing) with the required arguments :
   ```python
   import argparse
   from pathlib import Path

   parser = argparse.ArgumentParser()
   parser.add_argument("--train-dataset-path", required=True, type=Path)
   parser.add_argument("--test-dataset-path",  required=True, type=Path)
   parser.add_argument("--output-submission-path", required=False, type=Path, default=None)
   parser.add_argument("--checkpoint-path", required=False, type=Path, default=None)
   ```

   > `--output-submission-path` is optional (default `None`): only write the submission CSV if it is provided.
   > `--checkpoint-path` is optional (default `None`): used to resume training across multiple incremental runs (see § Checkpoint support below).

2. **Always include these standard hyperparameter arguments**, whether or not the original script has them. Use the original value as the default if present, otherwise use the sensible default shown below:
   ```python
   parser.add_argument("--gradient-steps", type=int,   default=<current_value or 10000>)
   parser.add_argument("--batch-size",     type=int,   default=<current_value or 256>)
   parser.add_argument("--lr",             type=float, default=<current_value or 1e-3>)
   parser.add_argument("--kfold",          type=int,   default=<current_value or 1>)
   ```

   > These arguments are injected by the agent at runtime via `additional_args`. A script that doesn't declare them will crash with `unrecognized arguments`. Even if the original script doesn't implement KFold, `--kfold` must be declared.

   > **`--kfold` implementation**: always implement GroupKFold, even if the original script only had a fixed internal split. When `--kfold 1`, fall back to a single 90/10 split (same as the original). When `--kfold > 1`, use `GroupKFold(n_splits=kfold)` grouped by `breath_id`, train on each fold, and average test predictions across folds. The val score reported as `Final Validation Score` is the mean MAE across all folds. This is the one allowed logic change during reformatting.

   > **`--gradient-steps`**: rename `--max-grad-steps`, `--num-steps`, `--n-steps` or similar to `--gradient-steps` for consistency. The training loop must respect this value.

   > **Keep all other existing hyperparameters** as optional arguments with their current default values. Example :
   ```

3. **Replace hardcoded paths** with the `args.*` variables :
   - Any read of the training dataset → `args.train_dataset_path`
   - Any read of the test dataset → `args.test_dataset_path`
   - Any write of the submission → `args.output_submission_path`

4. **Create the parent directory** before writing the submission :
   ```python
   args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
   submission.to_csv(args.output_submission_path, index=False)
   ```

5. **Wrap in `if __name__ == "__main__":`** if not already done.

---

### Special case: code with train + predict (no separate test)

If the original code uses an internal `train_test_split` to evaluate then predict on that same split, **you must adapt** as follows — this is the only allowed logic change, because the real test is provided separately :

**Before (pattern to replace) :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
model.fit(X_train, y_train)
preds = model.predict(X_test)
# submission write on internal X_test
```

**After :**
```python
# Train on ALL of train
model.fit(X, y)

# Load real test and predict
test_df = pd.read_csv(args.test_dataset_path)
# ... same preprocessing as train ...
preds = model.predict(X_test_real)
```

> Keep all preprocessing (encoding, scaling, feature engineering) — apply it to the real test the same way as to the train.

> If the split was only used to compute a local validation score, remove it (grading is done externally). If the split was used to select a model or for early stopping, keep that logic but train the final model on the full set.

---

### Submission format

The CSV written to `--output-submission-path` must have **exactly** the columns of the competition’s `sample_submission.csv`. If the original code does not match that format, align column names without changing the values.

---

---

### Checkpoint support

The agent calls the same script multiple times with increasing `--gradient-steps` (e.g. 1 000 → 5 000 → 10 000) to produce graded submissions at intermediate steps, enable early stopping, or resume after a crash. The script must support this correctly.

#### Arguments

```python
parser.add_argument("--gradient-steps", type=int, default=<current_value>)
parser.add_argument("--checkpoint-path", required=False, type=Path, default=None)
```

#### What to save

Save after training completes (or after each fold for KFold scripts):

```python
if args.checkpoint_path:
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "completed_steps":  args.gradient_steps,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),  # saved but NOT loaded on resume
        # single-fold scripts also save:
        "best_val_score":   best_val_score,
        "best_model_state": best_model_state,
    }, args.checkpoint_path)
```

For KFold scripts, save per-fold states and predictions:

```python
torch.save({
    "completed_steps":   args.gradient_steps,
    "fold_states":       {fold: {"model_state": ..., "optimizer_state": ..., "scheduler_state": ..., "scaler_state": ...}},
    "fold_test_preds":   {fold: np.array(...)},
    "fold_val_scores":   {fold: float(...)},
}, args.checkpoint_path)
```

#### What to load on resume — scheduler rule

**Always load** model state and optimizer state (preserves weights and momentum/LR).
**Never load** scheduler state: the scheduler is recreated fresh with `total_steps = args.gradient_steps`. This correctly handles incremental runs where `--gradient-steps` increases between calls.

```python
if args.checkpoint_path and args.checkpoint_path.exists():
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # scheduler not restored: recreated with new total_steps for incremental resume
    start_step = ckpt["completed_steps"]
```

> **Why not the scheduler?** Schedulers like `OneCycleLR` have a fixed shape tied to `total_steps`. Loading a state saved for 1 000 steps into a scheduler created for 5 000 steps produces a corrupted LR curve. Instead, let the scheduler restart its curve over the remaining steps — the LR from the optimizer state is already correct, so the first few steps are a minor warmup perturbation, negligible for long runs.

#### Incremental resume for KFold scripts

When the checkpoint was saved with `completed_steps=1000` but the new run targets `gradient_steps=5000`, all folds must be retrained from step 1 000 to 5 000:

```python
start_step = ckpt.get("completed_steps", 0)
fold_states = dict(ckpt.get("fold_states", {}))

if start_step >= args.gradient_steps:
    # Already at target — use cached fold results directly
    completed_folds_in_ckpt = <nb folds in checkpoint>
    # load fold_test_preds and fold_val_scores from ckpt
else:
    # Incremental: retrain all folds from start_step to gradient_steps
    completed_folds_in_ckpt = 0
    # reset fold_test_preds and fold_val_scores accumulators

for fold in range(args.kfold):
    if fold < completed_folds_in_ckpt:
        continue
    # create model + optimizer + scheduler fresh
    if fold in fold_states:
        model.load_state_dict(fold_states[fold]["model_state"])
        optimizer.load_state_dict(fold_states[fold]["optimizer_state"])
        # do NOT load scheduler_state
    step = start_step
    while step < args.gradient_steps:
        ...
```

#### Printing the final validation score

The agent parses the stdout to detect early stopping progress. Always print this line at the very end:

```python
print(f"Final Validation Score: {best_val_score}")
```

The value must be the **validation MAE** (lower is better), computed on the held-out validation set (not the test set).

---

### What you must NOT do

- Change the algorithm or model used
- Modify default hyperparameters
- Add feature engineering not present in the original code
- Refactor or reorganize the code beyond what is strictly necessary
- Add unnecessary imports
- Change preprocessing logic

---

## Full example of expected output

```python
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--train-dataset-path",      required=True, type=Path)
parser.add_argument("--test-dataset-path",        required=True, type=Path)
parser.add_argument("--output-submission-path",   required=True, type=Path)
parser.add_argument("--n-estimators", type=int,   default=100)
parser.add_argument("--max-depth",    type=int,   default=None)
args = parser.parse_args()

if __name__ == "__main__":
    train = pd.read_csv(args.train_dataset_path)
    test  = pd.read_csv(args.test_dataset_path)

    # ... feature engineering identical to the original ...
    X      = train.drop(columns=["target"])
    y      = train["target"]
    X_test = test.drop(columns=["id"])

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model.fit(X, y)
    preds = model.predict(X_test)

    submission = pd.DataFrame({"id": test["id"], "target": preds})
    args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_submission_path, index=False)
```
