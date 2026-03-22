#!/usr/bin/env python3
"""
analyze_result.py — Generate an HTML report from a custom-agent experiment folder.

Usage:
    python analyze_result.py /path/to/exp_folder
    python analyze_result.py /path/to/exp_folder --output my_report.html
"""

import argparse
import json
import re
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate HTML report from experiment folder")
parser.add_argument("folder", type=Path, help="Experiment folder (absolute path)")
parser.add_argument("--output", type=Path, default=None,
                    help="Output HTML path (default: results_<exp_id>.html next to folder)")
parser.add_argument("--no-open", action="store_true",
                    help="Do not open the report in the browser after writing")
args = parser.parse_args()

folder = args.folder.expanduser().resolve()
if not folder.exists():
    raise SystemExit(f"Folder not found: {folder}")

# ─────────────────────────────────────────────────────────────
# Discover files
# ─────────────────────────────────────────────────────────────
metric_files = sorted(folder.rglob("metric_*.json"))
run_info_files = list(folder.rglob("run_info.json"))

if not metric_files:
    raise SystemExit(f"No metric_*.json files found in {folder}")

# ─────────────────────────────────────────────────────────────
# Parse metrics
# ─────────────────────────────────────────────────────────────
records = []
for f in metric_files:
    try:
        d = json.loads(f.read_text())
        records.append(d)
    except Exception as e:
        print(f"Warning: could not parse {f.name}: {e}")

# ─────────────────────────────────────────────────────────────
# Parse run_info
# ─────────────────────────────────────────────────────────────
run_info = {}
if run_info_files:
    try:
        run_info = json.loads(run_info_files[0].read_text())
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# Experiment ID
# ─────────────────────────────────────────────────────────────
exp_id = run_info.get("experiment_id") or ""
if not exp_id:
    m = re.search(r'(exp_[A-Za-z0-9]+)', str(folder))
    exp_id = m.group(1) if m else folder.name

# ─────────────────────────────────────────────────────────────
# Group records by script, sorted by gradient_steps
# ─────────────────────────────────────────────────────────────
by_script = defaultdict(list)        # full checkpoint records (have grade)
by_script_patience = defaultdict(list)  # patience-only records (val_score only)

for r in records:
    name = Path(r.get("python_file", "")).stem or r.get("script_name", "unknown")
    if r.get("is_patience_only"):
        by_script_patience[name].append(r)
    else:
        by_script[name].append(r)

for name in by_script:
    by_script[name].sort(key=lambda x: x.get("gradient_steps", 0))
for name in by_script_patience:
    by_script_patience[name].sort(key=lambda x: x.get("gradient_steps", 0))

scripts = sorted(set(list(by_script.keys()) + list(by_script_patience.keys())))

# ─────────────────────────────────────────────────────────────
# Summary stats (full records only)
# ─────────────────────────────────────────────────────────────
full_records = [r for r in records if not r.get("is_patience_only")]
all_scored = [(r["score"], Path(r.get("python_file","")).stem, r.get("gradient_steps"))
              for r in full_records if r.get("score") is not None]
best = min(all_scored, key=lambda x: x[0]) if all_scored else (None, "N/A", None)
total_time_s = sum(r.get("execution_time_seconds", 0) for r in full_records)
early_stopped_scripts = {Path(r.get("python_file","")).stem
                          for r in full_records if r.get("early_stopped")}

# ─────────────────────────────────────────────────────────────
# Config from run_info or inferred from cmd
# ─────────────────────────────────────────────────────────────
train_dataset = run_info.get("train_dataset_path", "")
train_dataset_size: Optional[int] = run_info.get("train_dataset_size")
train_n_sequences: Optional[int] = run_info.get("train_n_sequences")  # breath_ids if sequence model
additional_args = run_info.get("additional_args", [])
checkpoint_steps = run_info.get("checkpoint_steps") or []
start_time = run_info.get("start_time", "")
end_time = run_info.get("end_time", "")
parallelism = run_info.get("parallelism")       # None if run_info absent
checkpoint_order = run_info.get("checkpoint_order")  # None if run_info absent

# Try to read dataset size from disk if not in run_info
if train_dataset_size is None and train_dataset:
    try:
        with open(train_dataset, "rb") as _f:
            train_dataset_size = sum(1 for _ in _f) - 1
    except Exception:
        pass

def _extract_arg(cmd_list, key):
    try:
        idx = cmd_list.index(key)
        return str(cmd_list[idx + 1])
    except (ValueError, IndexError):
        return None

# Infer args from first cmd in records
_sample_cmd = next((r["cmd"] for r in records if r.get("cmd")), [])
_sample_cmd_str = [str(x) for x in _sample_cmd]
inferred = {}
for key in ["--kfold", "--lr", "--batch-size", "--gradient-steps"]:
    v = _extract_arg(_sample_cmd_str, key)
    if v:
        inferred[key.lstrip("-")] = v
# Also from additional_args
_add_str = [str(a) for a in additional_args]
for key in ["--kfold", "--lr", "--batch-size", "--gradient-steps"]:
    if key.lstrip("-") not in inferred:
        v = _extract_arg(_add_str, key)
        if v:
            inferred[key.lstrip("-")] = v

# Epoch computation helper
_batch_size = int(inferred.get("batch-size", 0)) or None
_kfold      = int(inferred.get("kfold", 1)) or 1

# Use sequence count (breaths) if available, else row count
# sequence model: 1 step = batch_size sequences; row model: 1 step = batch_size rows
_epoch_base = train_n_sequences if train_n_sequences else train_dataset_size
_epoch_label = "seq" if train_n_sequences else "row"

def steps_to_epochs(steps: int) -> Optional[float]:
    """epochs = steps * batch_size / train_samples_per_fold
    For sequence models (breath_id): samples = n_sequences * train_fraction
    For row models: samples = n_rows * train_fraction"""
    if not _batch_size or not _epoch_base:
        return None
    if _kfold > 1:
        train_samples = _epoch_base * (_kfold - 1) / _kfold
    else:
        train_samples = _epoch_base * 0.9
    return round(steps * _batch_size / train_samples, 2)

def fmt_epochs(steps) -> str:
    e = steps_to_epochs(steps)
    return f"{e:.1f}" if e is not None else "—"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def fmt_time(seconds):
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def fmt_score(v):
    return f"{v:.4f}" if v is not None else "—"

# ─────────────────────────────────────────────────────────────
# Plotly traces
# ─────────────────────────────────────────────────────────────
COLORS = [
    "#58a6ff", "#f85149", "#3fb950", "#d29922", "#bc8cff",
    "#79c0ff", "#ff7b72", "#56d364", "#e3b341", "#ff9bce",
]
def color(i): return COLORS[i % len(COLORS)]

mae_traces = []
val_traces  = []
time_traces = []

for i, script in enumerate(scripts):
    recs      = by_script.get(script, [])
    p_recs    = by_script_patience.get(script, [])
    steps     = [r.get("gradient_steps", 0) for r in recs]
    grades    = [r.get("score") for r in recs]
    times     = [r.get("execution_time_seconds", 0) for r in recs]
    label     = script

    # Val trace: merge full-checkpoint val + patience-only val, sorted by step
    val_combined = sorted(
        [(r.get("gradient_steps", 0), r.get("val_score")) for r in recs + p_recs
         if r.get("val_score") is not None],
        key=lambda x: x[0],
    )
    val_steps  = [v[0] for v in val_combined]
    val_scores = [v[1] for v in val_combined]
    # Mark patience-only steps as open circles, full steps as filled
    patience_step_set = {r.get("gradient_steps") for r in p_recs}
    val_symbols = ["circle-open" if s in patience_step_set else "circle" for s in val_steps]

    epochs_for_steps = [steps_to_epochs(s) for s in steps]
    grade_hover = [
        f"step {s}<br>epoch {f'{e:.1f}' if e else '?'}<br>MAE {f'{g:.4f}' if g else '—'}"
        for s, e, g in zip(steps, epochs_for_steps, grades)
    ]
    epochs_for_val = [steps_to_epochs(s) for s in val_steps]
    val_hover = [
        f"step {s}<br>epoch {f'{e:.1f}' if e else '?'}<br>val {f'{v:.4f}' if v else '—'}"
        for s, e, v in zip(val_steps, epochs_for_val, val_scores)
    ]

    if recs:
        mae_traces.append({
            "type": "scatter", "x": steps, "y": grades,
            "name": label, "mode": "lines+markers",
            "line": {"color": color(i), "width": 2},
            "marker": {"size": 8}, "legendgroup": script,
            "text": grade_hover, "hovertemplate": "%{text}<extra>%{fullData.name}</extra>",
        })
    if val_combined:
        val_traces.append({
            "type": "scatter", "x": val_steps, "y": val_scores,
            "name": label + " (val)", "mode": "lines+markers",
            "line": {"color": color(i), "width": 1, "dash": "dash"},
            "marker": {"size": 5, "symbol": val_symbols},
            "legendgroup": script, "showlegend": True,
            "text": val_hover, "hovertemplate": "%{text}<extra>%{fullData.name}</extra>",
        })
    if recs:
        time_traces.append({
            "type": "bar", "x": [f"{s}" for s in steps], "y": times,
            "name": label, "marker": {"color": color(i)},
            "legendgroup": script,
        })

mae_layout = {
    "title": "MAE over gradient steps  (solid = grade, dashed = val)",
    "xaxis": {"title": "Gradient Steps"},
    "yaxis": {"title": "MAE (lower is better)"},
    "template": "plotly_dark", "hovermode": "x unified",
    "legend": {"orientation": "h", "y": -0.25},
    "height": 480, "margin": {"b": 120},
}
time_layout = {
    "title": "Execution time per invocation (seconds)",
    "xaxis": {"title": "Gradient Steps"},
    "yaxis": {"title": "Seconds"},
    "template": "plotly_dark", "barmode": "group",
    "legend": {"orientation": "h", "y": -0.3},
    "height": 360, "margin": {"b": 120},
}

# ─────────────────────────────────────────────────────────────
# Stop reasons table
# ─────────────────────────────────────────────────────────────
stop_rows = []
for i, script in enumerate(scripts):
    recs = by_script.get(script, [])
    if not recs:
        continue
    stopped = any(r.get("early_stopped") for r in recs)
    stop_step = next((r.get("gradient_steps") for r in recs if r.get("early_stopped")), None)
    max_step  = max((r.get("gradient_steps", 0) for r in recs), default=0)
    best_g    = min((r["score"] for r in recs if r.get("score") is not None), default=None)
    best_v    = min((r["val_score"] for r in recs if r.get("val_score") is not None), default=None)
    total_t   = sum(r.get("execution_time_seconds", 0) for r in recs)
    n_runs    = len(recs)

    if stopped:
        reason = f'<span class="badge-early">Patience @ step {stop_step}</span>'
    else:
        reason = f'<span class="badge-ok">Completed @ step {max_step}</span>'

    dot = f'<span style="color:{color(i)}">●</span>'
    stop_rows.append(
        f"<tr>"
        f"<td>{dot} {script}</td>"
        f"<td>{reason}</td>"
        f"<td>{fmt_score(best_g)}</td>"
        f"<td>{fmt_score(best_v)}</td>"
        f"<td>{n_runs}</td>"
        f"<td>{fmt_time(total_t)}</td>"
        f"</tr>"
    )

# ─────────────────────────────────────────────────────────────
# Detailed results table
# ─────────────────────────────────────────────────────────────
detail_rows = []
for i, script in enumerate(scripts):
    recs = by_script.get(script, [])
    if not recs:
        continue
    for r in recs:
        score = r.get("score")
        val   = r.get("val_score")
        err   = r.get("error") or ""
        es    = r.get("early_stopped", False)

        diff_str = "—"
        if score is not None and val is not None:
            d = score - val
            diff_str = f"{'+'if d>=0 else ''}{d:.4f}"

        if err:
            status = '<span class="badge-err">ERROR</span>'
        elif es:
            status = '<span class="badge-early">EARLY STOP</span>'
        else:
            status = '<span class="badge-ok">OK</span>'

        steps_val = r.get('gradient_steps', 0)
        dot = f'<span style="color:{color(i)}">●</span>'
        detail_rows.append(
            f"<tr>"
            f"<td>{dot} {script}</td>"
            f"<td>{steps_val}</td>"
            f"<td style='color:#8b949e'>{fmt_epochs(steps_val)}</td>"
            f"<td>{fmt_score(score)}</td>"
            f"<td>{fmt_score(val)}</td>"
            f"<td>{diff_str}</td>"
            f"<td>{fmt_time(r.get('execution_time_seconds',0))}</td>"
            f"<td>{status}</td>"
            f"<td class='err-msg'>{err[:120] if err else ''}</td>"
            f"</tr>"
        )

# ─────────────────────────────────────────────────────────────
# Summary block
# ─────────────────────────────────────────────────────────────
best_score_str  = fmt_score(best[0])
best_script_str = f"{best[1]}@step{best[2]}" if best[0] is not None else "N/A"

args_parts = []
if train_dataset:
    args_parts.append(f"<b>dataset:</b> {Path(train_dataset).name}")
for k, v in inferred.items():
    args_parts.append(f"<b>{k}:</b> {v}")
if checkpoint_steps:
    args_parts.append(f"<b>checkpoint-steps:</b> {checkpoint_steps}")
elif full_records:
    _inferred_ckpt = sorted({r.get("gradient_steps") for r in full_records if r.get("gradient_steps")})
    if _inferred_ckpt:
        args_parts.append(f"<b>checkpoint-steps:</b> {_inferred_ckpt} (inferred)")
if run_info.get("patience_every"):
    args_parts.append(f"<b>patience-every:</b> {run_info['patience_every']}")
elif by_script_patience:
    _all_p_steps = sorted({r.get("gradient_steps") for recs in by_script_patience.values() for r in recs if r.get("gradient_steps")})
    if _all_p_steps:
        _gaps = sorted({_all_p_steps[i+1] - _all_p_steps[i] for i in range(len(_all_p_steps)-1)})
        _gap_str = str(_gaps[0]) if len(_gaps) == 1 else str(_all_p_steps)
        args_parts.append(f"<b>patience-every:</b> ~{_gap_str} (inferred)")
if run_info.get("early_stopping_patience"):
    args_parts.append(f"<b>early-stopping-patience:</b> {run_info['early_stopping_patience']}")
if parallelism is not None:
    args_parts.append(f"<b>parallelism:</b> {parallelism}")
if checkpoint_order is not None:
    args_parts.append(f"<b>checkpoint-order:</b> {checkpoint_order}")
args_html = " &nbsp;|&nbsp; ".join(args_parts) if args_parts else "N/A"

duration_str = ""
if start_time and end_time:
    try:
        dt = datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
        duration_str = f" ({fmt_time(dt.total_seconds())} wall clock)"
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# Render HTML
# ─────────────────────────────────────────────────────────────
traces_json  = json.dumps(mae_traces + val_traces)
mae_lay_json = json.dumps(mae_layout)
time_json    = json.dumps(time_traces)
time_lay_json = json.dumps(time_layout)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — {exp_id}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 24px; font-size: 14px; }}
  h1 {{ color: #58a6ff; font-size: 1.3em; margin-bottom: 4px; }}
  h2 {{ color: #79c0ff; font-size: 1em; margin: 0 0 14px 0; border-bottom: 1px solid #21262d; padding-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }}
  .subtitle {{ color: #8b949e; font-size: 0.85em; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .stat {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }}
  .stat-label {{ font-size: 0.72em; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .stat-value {{ font-size: 1.05em; color: #e6edf3; word-break: break-all; }}
  .stat-value.best {{ color: #3fb950; font-weight: bold; font-size: 1.2em; }}
  .args-bar {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; font-size: 0.85em; line-height: 1.8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.83em; }}
  th {{ background: #21262d; color: #8b949e; padding: 8px 10px; text-align: left; font-weight: normal; text-transform: uppercase; font-size: 0.72em; letter-spacing: 1px; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }}
  tr:hover td {{ background: #1c2128; }}
  .badge-ok    {{ color: #3fb950; }}
  .badge-err   {{ color: #f85149; }}
  .badge-early {{ color: #d29922; }}
  .err-msg {{ font-size: 0.75em; color: #8b949e; max-width: 280px; word-break: break-all; }}
</style>
</head>
<body>

<h1>Experiment Results</h1>
<div class="subtitle">{exp_id} &nbsp;|&nbsp; {start_time} → {end_time}{duration_str}</div>

<div class="card">
  <h2>Summary</h2>
  <div class="grid4">
    <div class="stat">
      <div class="stat-label">Best Grade MAE</div>
      <div class="stat-value best">{best_score_str}</div>
      <div style="font-size:0.75em;color:#8b949e;margin-top:4px">{best_script_str}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Total execution time</div>
      <div class="stat-value">{fmt_time(total_time_s)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Scripts × Checkpoints</div>
      <div class="stat-value">{len(scripts)} × {len(set(r.get("gradient_steps") for r in full_records))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Early stopped</div>
      <div class="stat-value {'badge-early' if early_stopped_scripts else ''}">{len(early_stopped_scripts)} / {len(scripts)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Dataset</div>
      <div class="stat-value">{f"{train_dataset_size:,} rows" if train_dataset_size else "—"}</div>
      <div style="font-size:0.75em;color:#8b949e;margin-top:2px">{f"{train_n_sequences:,} sequences (breath_id)" if train_n_sequences else ""}</div>
      <div style="font-size:0.75em;color:#8b949e;margin-top:2px">epochs based on <b>{_epoch_label}s</b></div>
      <div style="font-size:0.75em;color:#8b949e;margin-top:2px">{Path(train_dataset).name if train_dataset else ""}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Batch size</div>
      <div class="stat-value">{_batch_size if _batch_size else "—"}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Learning rate</div>
      <div class="stat-value">{inferred.get("lr", "—")}</div>
    </div>
    <div class="stat">
      <div class="stat-label">KFold</div>
      <div class="stat-value">{inferred.get("kfold", "—")}</div>
    </div>
  </div>
  <div class="args-bar">{args_html}</div>
</div>

<div class="card">
  <h2>Per-script summary</h2>
  <table>
    <thead>
      <tr><th>Script</th><th>Stop reason</th><th>Best grade MAE</th><th>Best val MAE</th><th>Runs</th><th>Total time</th></tr>
    </thead>
    <tbody>{''.join(stop_rows)}</tbody>
  </table>
</div>

<div class="card">
  <h2>MAE over gradient steps</h2>
  <div id="mae-chart"></div>
</div>

<div class="card">
  <h2>Execution time per invocation</h2>
  <div id="time-chart"></div>
</div>

<div class="card">
  <h2>Detailed results</h2>
  <table>
    <thead>
      <tr><th>Script</th><th>Steps</th><th>Epochs</th><th>Grade MAE</th><th>Val MAE</th><th>Diff (grade−val)</th><th>Time</th><th>Status</th><th>Error</th></tr>
    </thead>
    <tbody>{''.join(detail_rows)}</tbody>
  </table>
</div>

<script>
Plotly.newPlot('mae-chart',  {traces_json},  {mae_lay_json},  {{responsive: true}});
Plotly.newPlot('time-chart', {time_json}, {time_lay_json}, {{responsive: true}});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────
if args.output:
    out_path = args.output
else:
    out_path = folder.parent / f"results_{exp_id}.html"

out_path.write_text(html, encoding="utf-8")
print(f"Report written: {out_path}")
print(f"  {len(records)} runs across {len(scripts)} script(s)")
if best[0] is not None:
    print(f"  Best grade MAE: {best[0]:.4f} ({best_script_str})")

if not args.no_open:
    webbrowser.open(out_path.as_uri())
