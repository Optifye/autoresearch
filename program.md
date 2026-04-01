# autoresearch

This is an experiment to have the agent do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current default branch.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: check that the required local cache exists (by default `~/.cache/autoresearch/onemed_dense_v1/`, unless the repo clearly specifies another path). If a local override is supported, use the repo’s existing mechanism such as `AUTORESEARCH_CACHE_DIR` or `--cache-dir`. For this branch, use only the single subassembly cache. Do not prepare or reference a button cache. The canonical prepare path is:

```bash
AUTORESEARCH_WORKSPACE_ROOT=/home/ubuntu/internvideo-attention \
AUTORESEARCH_ENV_PATH=/home/ubuntu/autoresearch/.env \
uv run python prepare.py \
  --cache-dir /tmp/autoresearch-minda-subassembly-cache \
  --run-id 92c8fdb4-c0f6-4503-b2cc-ab340f79f8f6 \
  --space-id a5fe549e-75fe-4cb3-80ce-9b60e33b89fb \
  --run-number 5 \
  --split-policy camera_stratified_hash \
  --val-ratio 0.4 \
  --force
```

If `/tmp/autoresearch-minda-subassembly-cache/onemed_dense_v1` already exists, reuse it and do not rerun `prepare.py`.
5. **Initialize `results.tsv`**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once setup is confirmed, kick off the experimentation.

## Experimentation

**Execution environment**

* Use this repo’s `uv` environment. The canonical runtime is `uv run python ...`.
* Prefer the existing checked-out workspace plus the repo’s own `.venv`; do not create ad hoc alternate environments.
* If the run needs shared checkpoints or credentials from another workspace, use the repo’s existing path-based hooks such as `AUTORESEARCH_WORKSPACE_ROOT` and `AUTORESEARCH_ENV_PATH`.

Each experiment runs on a single GPU. The training script runs for a **fixed 10-minute time budget**. For this branch, launch it as:

```bash
AUTORESEARCH_CACHE_DIR=/tmp/autoresearch-minda-subassembly-cache \
CUDA_VISIBLE_DEVICES=0 \
uv run python train.py
```

This branch reads the source run from the prepared cache manifest; do not pass button-specific env vars.

Treat the task-specific cache contract as fixed. In this standalone dense-temporal / vision setting, that means things like:

* fixed prepared cache / shards,
* fixed train/val split,
* fixed prepare/materialization path,
* fixed held-out validation protocol,
* fixed frozen upstream V-JEPA encoder assumptions unless `train.py` already makes something tunable.
* For this branch, the fixed contract is a single subassembly dataset with a deterministic `60/40` `camera_stratified_hash` validation split.

**What you CAN do:**

* Modify `train.py` only - this is the only file you edit
* Change architecture.
* Change temporal model.
* Change temporal model internals.
* Change optimizer, hyperparameters, training loop, batch size, model size, and related training details. But if incremental hyperparameter tuning starts to plateau, do not get stuck there; the bigger remaining gains are likely to come from systemic architectural changes to the network.
* Change losses and weighting.
* Use cached encoder tokens instead of pooled embeddings.
* Tune the pooler from `train.py`.
* Add auxiliary objectives, self-supervised techniques, LoRA, or architectural improvements as long as they stay within the same base V-JEPA setup.
* Add entirely new downstream temporal models.
* Simplify the code if it preserves or improves results.

**What you CANNOT do:**

* Modify `prepare.py` or the fixed data/evaluation harness.
* Modify the evaluation metric or how it is computed.
* Modify the train/val split.
* Add new packages or dependencies.
* Change the problem definition just to make the metric look better.
* Rewrite the repo into a totally different system if the experiment is meant to improve the existing stack rather than replace it.

**The goal is simple: improve the primary validation metric defined by the repo.**
For this single-space Minda subassembly branch, compare candidates by `val_pair_f1` on the held-out subassembly validation split, where `val_pair_f1` is the proxy-threshold macro-F1 over raw start/end logits (`prob=0.15`, `tolerance=2` windows). Tie-break with `val_proxy_total_false_count`, then `val_legacy_pair_f1`, then `val_count_mae`, then average timing MAE.

Since the runtime budget is fixed, you do not need to obsess over absolute training duration inside that budget. Everything is fair game within `train.py`: architecture, optimizer, losses, schedules, batching, parameterization, temporal context, decoding head, and representation consumption.

**Tie-breakers** matter. All else equal:

1. better primary metric first,
2. then better secondary metrics explicitly printed by the evaluator,
3. then simpler code.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: all else being equal, simpler is better. A tiny improvement that adds ugly, brittle complexity is often not worth it. Conversely, removing complexity and getting equal or better results is a win. When judging whether to keep a change, weigh complexity cost against improvement magnitude. A very small gain from a large pile of hacky code is probably not worth keeping. A similar gain from cleaner code, or from deleting code, is much more valuable.

**The first run**: your very first run should always be to establish the baseline, so run the training script exactly as-is.

After the baseline run, everything inside the allowed file is fair game immediately. There is no required search order. Small local improvements, loss changes, temporal model changes, and more radical ideas are all valid once the baseline exists.

## Output format

When the script finishes it should print a summary block like this:

```text
---
val_pair_f1:        0.000000
val_legacy_pair_f1: 0.000000
val_count_mae:      0.000000
val_start_mae_ms:   0.0
val_end_mae_ms:     0.0
training_seconds:   600.0
total_seconds:      620.0
time_budget_seconds:600.0
tcn_stage_seconds:  300.0
probe_stage_seconds:300.0
peak_vram_mb:       0.0
cache_version:      onemed_dense_v1
model_family:       minda_subassembly_mar13_stage0_historical_probe
task_mode:          boundary_pairs_single_space
pooler_tune_mode:   phase1_historical
representation_mode:pooled_z0_then_tokens
```

The exact fields may differ by repo and task mode, but the script must print a machine-readable summary block at the end. The primary metric and memory usage must be extractable from the log.

You can extract the key metrics from the log file with commands like:

```bash
grep "^val_pair_f1:\|^val_legacy_pair_f1:\|^val_count_mae:\|^val_start_mae_ms:\|^val_end_mae_ms:\|^peak_vram_mb:" run.log
```

If the summary block is missing, the run failed.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break descriptions).

The TSV should have a header row and these columns:

```text
commit	primary_metric	aux_metric	memory_gb	status	description
```

Where:

1. `commit` = git commit hash (short, 7 chars)
2. `primary_metric` = `val_pair_f1` on the held-out subassembly validation split (proxy-threshold macro-F1) — use `0.000000` for crashes
3. `aux_metric` = `val_count_mae`
4. `memory_gb` = peak memory in GB, round to `.1f` (divide `peak_vram_mb` by 1024) — use `0.0` for crashes
5. `status` = `keep`, `discard`, or `crash`
6. `description` = short text description of what this experiment tried

Use timing MAE as the next tie-breaker when `primary_metric` and `aux_metric` are effectively tied.

Example:

```text
commit	primary_metric	aux_metric	memory_gb	status	description
a1b2c3d	0.812300	0.440000	18.6	keep	subassembly baseline
b2c3d4e	0.826700	0.804000	18.9	keep	increase temporal receptive field
c3d4e5f	0.821000	0.781000	18.7	discard	switch auxiliary loss weighting
d4e5f6g	0.000000	0.000000	0.0	crash	double hidden width caused OOM
```

Do **not** commit `results.tsv`. Leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch / commit you are on.
2. Tune `train.py` with one concrete experimental idea by directly hacking the code.
3. git commit
4. Run the experiment:

```bash
AUTORESEARCH_CACHE_DIR=/tmp/autoresearch-minda-subassembly-cache \
CUDA_VISIBLE_DEVICES=0 \
uv run python train.py > run.log 2>&1
```

Redirect everything — do **NOT** use `tee` or let output flood your context.

5. Read out the results:

```bash
grep "^val_pair_f1:\|^val_legacy_pair_f1:\|^val_count_mae:\|^val_start_mae_ms:\|^val_end_mae_ms:\|^peak_vram_mb:" run.log
```

6. If the grep output is empty, the run crashed. Read the traceback with:

```bash
tail -n 80 run.log
```

Try to diagnose whether it was:

* a simple implementation bug,
* an OOM / resource issue,
* or a fundamentally bad idea.

7. Record the results in `results.tsv` (NOTE: do not commit the `results.tsv` file).
8. If the primary metric improved, you “advance” the branch, keeping the git commit.
9. If the primary metric is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them. You are advancing the branch so that good ideas compound over time. If you feel stuck, you may occasionally rewind or revisit older directions, but do this very very sparingly.

**Timeout**: each experiment should complete within the repo’s intended budget plus startup/eval overhead. In this repo the steady-state training budget is 10 minutes, so a run taking meaningfully beyond that (for example over 15 minutes total wall time) should be killed and treated as a failure.

**Crashes**: if a run crashes (OOM, bug, bad tensor shape, etc.), use your judgment:

* if it is something dumb and easy to fix (typo, import, shape bug, obvious config issue), fix it and rerun;
* if the idea itself is fundamentally broken, log `crash` in the TSV, revert, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user may leave you running while they sleep. If each experiment takes around the fixed budget, you can complete many experiments in one unattended stretch, leaving the human with a branch that has advanced and a `results.tsv` full of evidence rather than guesses. The user then wakes up to experimental results, all completed by you while they slept!
