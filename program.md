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
4. **Verify data exists**: check that the required local cache exists (by default `/tmp/autoresearch-mangalam-packing-cache/mangalam_dense_v1`, unless the repo clearly specifies another path). If a local override is supported, use the repo’s existing mechanism such as `AUTORESEARCH_CACHE_DIR` or `--cache-dir`. For this branch, use only the single Mangalam packing cache and the fixed pregenerated prod stage-0 checkpoint. The canonical prepare path is:

```bash
AUTORESEARCH_WORKSPACE_ROOT=/home/ubuntu/internvideo-attention \
AUTORESEARCH_ENV_PATH=/home/ubuntu/autoresearch/.env \
uv run python prepare.py \
  --cache-dir /tmp/autoresearch-mangalam-packing-cache \
  --run-id 9713de5f-df45-49b7-9b41-24fb768a6325 \
  --space-id 1e2544ee-b324-4c86-8453-e1fd7fad9c04 \
  --run-number 10 \
  --split-policy camera_stratified_hash \
  --val-ratio 0.4 \
  --force
```

If `/tmp/autoresearch-mangalam-packing-cache/mangalam_dense_v1` already exists, reuse it and do not rerun `prepare.py`.
Also verify the fixed prod stage-0 checkpoint exists at `/tmp/autoresearch-mangalam-packing-cache/fixed_stage0_prod_reduced_rf_halo16/boundary_model.pt`, unless `AUTORESEARCH_FIXED_STAGE0_CHECKPOINT` points elsewhere.
5. **Initialize `results.tsv`**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once setup is confirmed, kick off the experimentation.

## Experimentation

**Execution environment**

- Use this repo’s `uv` environment. The canonical runtime is `uv run python ...`.
- Prefer the existing checked-out workspace plus the repo’s own `.venv`; do not create ad hoc alternate environments.
- If the run needs shared checkpoints or credentials from another workspace, use the repo’s existing path-based hooks such as `AUTORESEARCH_WORKSPACE_ROOT` and `AUTORESEARCH_ENV_PATH`.

Each experiment runs on a single GPU. The training script runs for a **fixed 5-minute time budget**. This branch is a **phase-1-only** research track: the TCN is a fixed pregenerated prod `reduced_rf_halo16` checkpoint, and `train.py` only runs pooler finetuning plus per-epoch validation. For this branch, launch it as:

```bash
AUTORESEARCH_CACHE_DIR=/tmp/autoresearch-mangalam-packing-cache \
AUTORESEARCH_FIXED_STAGE0_CHECKPOINT=/tmp/autoresearch-mangalam-packing-cache/fixed_stage0_prod_reduced_rf_halo16/boundary_model.pt \
AUTORESEARCH_TIME_BUDGET_SECONDS=300 \
AUTORESEARCH_TCN_STAGE_SECONDS=0 \
AUTORESEARCH_PROBE_STAGE_SECONDS=300 \
CUDA_VISIBLE_DEVICES=0 \
uv run python train.py
```

This branch reads the source run from the prepared cache manifest; do not pass button-specific env vars. Do not retrain stage 0 in this branch. The only trainable path is phase-1 pooler finetuning over the fixed prod baseline TCN.

Treat the task-specific cache contract as fixed. In this standalone dense-temporal / vision setting, that means things like:

- fixed prepared cache / shards,
- fixed train/val split,
- fixed prepare/materialization path,
- fixed held-out validation protocol,
- fixed frozen upstream V-JEPA encoder assumptions unless `train.py` already makes something tunable.
- fixed pregenerated prod `reduced_rf_halo16` stage-0 checkpoint.
- For this branch, the fixed contract is a single Mangalam packing dataset with a deterministic `60/40` `camera_stratified_hash` validation split.

**What you CAN do:**

- Modify `train.py` only - this is the only file you edit
- Change phase-1 optimizer, hyperparameters, training loop, batching, and schedules.
- Change pooler finetuning structure, regularization, losses, and sampling while staying within the fixed evaluation contract.
- Change how `train.py` wraps or replaces the baseline phase-1 runner, as long as the fixed cache, split, and pregenerated stage-0 checkpoint stay fixed.
- Simplify the code if it preserves or improves results.

**What you CANNOT do:**

- Modify `prepare.py` or the fixed data/evaluation harness.
- Modify the evaluation metric or how it is computed.
- Modify the train/val split.
- Add new packages or dependencies.
- Retrain or alter the fixed stage-0 TCN checkpoint inside this branch’s experiment loop.
- Change the fixed prod baseline recipe that the pregenerated stage-0 checkpoint is supposed to represent.
- Change the problem definition just to make the metric look better.
- Rewrite the repo into a totally different system if the experiment is meant to improve the existing stack rather than replace it.

Do not default to narrow hyperparameter optimization. The branch is at the point where the biggest remaining gains are more likely to come from better phase-1 supervision, better pooler finetuning schedules, better stream sampling, better regularization, or cleaner selection logic that still respects the fixed contract. Use hyperparameter tuning mainly to support, stabilize, or validate one of those larger ideas.

**The goal is simple: improve phase-1 validation halo16 quality without trading away the fixed-contract behavior.**
For this single-space Mangalam packing branch, compare candidates by `val_halo16_pair_f1` on the held-out `60/40` validation split. Tie-break with lower `val_proxy_total_false_count`, then lower `val_count_mae`, then lower timing MAE.

Since the runtime budget is fixed, you do not need to obsess over absolute training duration inside that budget. Everything is fair game within `train.py` so long as it stays inside the fixed 5-minute phase-1-only budget and keeps the fixed cache / split / stage-0 checkpoint contract intact.

**Tie-breakers** matter. All else equal:

1. better `val_halo16_pair_f1` first,
2. then lower `val_proxy_total_false_count`,
3. then lower `val_count_mae`,
4. then simpler code.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: all else being equal, simpler is better. A tiny improvement that adds ugly, brittle complexity is often not worth it. Conversely, removing complexity and getting equal or better results is a win. When judging whether to keep a change, weigh complexity cost against improvement magnitude. A very small gain from a large pile of hacky code is probably not worth keeping. A similar gain from cleaner code, or from deleting code, is much more valuable.

**The first run**: your very first run should always be to establish the baseline, so run the training script exactly as-is.

After the baseline run, everything inside the allowed file is fair game immediately. There is no required search order. Small local improvements, loss changes, temporal model changes, and more radical ideas are all valid once the baseline exists.

## Output format

When the script finishes it should print a summary block like this:

```text
---
val_halo16_pair_f1:0.000000
val_legacy_pair_f1:0.000000
val_chunk32_pair_f1:0.000000
val_proxy_macro_f1:0.000000
val_proxy_false:   0
val_count_mae:     0.000000
val_timing_mae_ms: 0.0
baseline_halo16_pair_f1:0.000000
best_epoch:        0
training_seconds:   300.0
total_seconds:      320.0
time_budget_seconds:300.0
tcn_stage_seconds:  0.0
probe_stage_seconds:300.0
peak_vram_mb:       0.0
cache_version:      mangalam_dense_v1
model_family:       mangalam_phase1_prod6040_halo16_fixed_stage0
task_mode:          pooler_phase1_single_space_fixed_stage0
pooler_tune_mode:   phase1_prod_v2_epoch_eval
representation_mode:tokens_to_pooler_fixed_stage0
```

The exact fields may differ by repo and task mode, but the script must print a machine-readable summary block at the end. The primary metric and memory usage must be extractable from the log.

You can extract the key metrics from the log file with commands like:

```bash
grep "^val_halo16_pair_f1:\|^val_proxy_macro_f1:\|^val_proxy_false:\|^val_count_mae:\|^val_timing_mae_ms:\|^peak_vram_mb:" run.log
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
2. `primary_metric` = `val_halo16_pair_f1` on the held-out Mangalam validation split — use `0.000000` for crashes
3. `aux_metric` = `val_proxy_total_false_count` — use `0.0` for crashes
4. `memory_gb` = peak memory in GB, round to `.1f` (divide `peak_vram_mb` by 1024) — use `0.0` for crashes
5. `status` = `keep`, `discard`, or `crash`
6. `description` = short text description of what this experiment tried; include the structural idea, and usually include the component metrics (`halo16`, `proxy`, `count_mae`) for auditability

Use `val_count_mae`, then timing MAE, as the next tie-breakers when `primary_metric` and `aux_metric` are effectively tied.

Example:

```text
commit	primary_metric	aux_metric	memory_gb	status	description
a1b2c3d	0.938800	205.0	18.6	keep	mangalam phase1 baseline halo16=0.9388 proxy=0.9022 count_mae=0.5000
b2c3d4e	0.942100	169.0	18.9	keep	change stream sampling halo16=0.9421 proxy=0.9040 count_mae=0.4286
c3d4e5f	0.934500	188.0	18.7	discard	switch distill weighting halo16=0.9345 proxy=0.8991 count_mae=0.5714
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
AUTORESEARCH_CACHE_DIR=/tmp/autoresearch-mangalam-packing-cache \
AUTORESEARCH_FIXED_STAGE0_CHECKPOINT=/tmp/autoresearch-mangalam-packing-cache/fixed_stage0_prod_reduced_rf_halo16/boundary_model.pt \
AUTORESEARCH_TIME_BUDGET_SECONDS=300 \
AUTORESEARCH_TCN_STAGE_SECONDS=0 \
AUTORESEARCH_PROBE_STAGE_SECONDS=300 \
CUDA_VISIBLE_DEVICES=0 \
uv run python train.py > run.log 2>&1
```

Redirect everything — do **NOT** use `tee` or let output flood your context.

1. Read out the results:

```bash
grep "^val_halo16_pair_f1:\|^val_proxy_macro_f1:\|^val_proxy_false:\|^val_count_mae:\|^val_timing_mae_ms:\|^peak_vram_mb:" run.log
```

1. If the grep output is empty, the run crashed. Read the traceback with:

```bash
tail -n 80 run.log
```

Try to diagnose whether it was:

- a simple implementation bug,
- an OOM / resource issue,
- or a fundamentally bad idea.

1. Record the results in `results.tsv` (NOTE: do not commit the `results.tsv` file).
2. If the primary metric improved, you “advance” the branch, keeping the git commit.
3. If the primary metric is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them. You are advancing the branch so that good ideas compound over time. If you feel stuck, you may occasionally rewind or revisit older directions, but do this very very sparingly.

**Timeout**: each experiment should complete within the repo’s intended budget plus startup/eval overhead. In this repo the steady-state training budget is 5 minutes, so a run taking meaningfully beyond that (for example over 10 minutes total wall time) should be killed and treated as a failure.

**Crashes**: if a run crashes (OOM, bug, bad tensor shape, etc.), use your judgment:

- if it is something dumb and easy to fix (typo, import, shape bug, obvious config issue), fix it and rerun;
- if the idea itself is fundamentally broken, log `crash` in the TSV, revert, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user may leave you running while they sleep. If each experiment takes around the fixed budget, you can complete many experiments in one unattended stretch, leaving the human with a branch that has advanced and a `results.tsv` full of evidence rather than guesses. The user then wakes up to experimental results, all completed by you while they slept!
