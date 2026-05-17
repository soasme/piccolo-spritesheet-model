# piccolo-spritesheet autoresearch

This repo is set up for autonomous experiments on a small JEPA sprite model.

The job is simple: improve `train.py` to get the lowest possible `pred_error` from `eval.py`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current baseline commit.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context and constraints.
   - `prepare.py` — data download and frame slicing. Do not modify.
   - `train.py` — the only file you optimize.
   - `eval.py` — evaluation entrypoint and source-of-truth metric. Do not modify.
4. **Verify data**: Check that `data/frames/` exists and has frame pairs. If not, run `uv run --no-sync prepare.py`. If network is unavailable, ask the human.
5. **Verify dependencies**: Run `uv sync` if needed, then reinstall the CUDA-compatible torch on the server: `uv pip install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124`.
6. **Pick device**: `cuda` if available, `mps` on Apple Silicon, otherwise `cpu`.
7. **Initialize results.tsv**: Create it with just the header row if it does not exist. **Do NOT commit results.tsv** — leave it untracked by git.

```tsv
commit	pred_error	memory_gb	status	description
```

Once setup is confirmed, kick off the experimentation.

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes** (wall-clock training time, excluding startup). Launch as:

```bash
uv run --no-sync train.py > run.log 2>&1
uv run --no-sync eval.py --device <device> > eval.log 2>&1
```

### What you CAN do

- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, loss weighting, optimizer, scheduler, batch size, latent size, predictor design, logging, checkpointing, and training loop details.

### What you CANNOT do

- Modify `prepare.py` or `eval.py`.
- Change the dataset contents by hand.
- Add new dependencies.
- Change the metric definition.

### Goal

**Minimize `pred_error`.** Lower is better. Training loss is only a proxy; `eval.py` is the source of truth.

### VRAM

VRAM is a soft constraint. Some increase is acceptable for meaningful `pred_error` gains, but it should not blow up dramatically.

### Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

### First run

The first run should always be the baseline with no code changes, to establish the starting point.

## Output format

Training prints progress lines and a final summary:

```
step      0  loss=0.1234  pred_error=0.5678  t=0s
...
---
training_seconds: 300.1
peak_vram_mb:     1234.5
num_steps:        2700
```

Evaluation prints the metric:

```
pred_error (mean cosine distance): 0.1234
```

Extract key values after a run:

```bash
grep "pred_error" eval.log
grep "^peak_vram_mb:" run.log
```

## Logging results

After every experiment, append one row to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

Columns:

1. `commit` — short git hash (7 chars) for kept/discarded runs; use `working` for uncommitted crash/debug runs.
2. `pred_error` — numeric metric from `eval.py`; use `0.000000` for crashes.
3. `memory_gb` — peak GPU memory in GB, rounded to one decimal (from `grep "^peak_vram_mb:" run.log`, divide by 1024); use `0.0` for crashes or CPU runs.
4. `status` — `keep`, `discard`, or `crash`.
5. `description` — short text description of what this experiment tried.

Example:

```tsv
commit	pred_error	memory_gb	status	description
4f2c1ab	0.412300	1.2	keep	baseline
9a8b7cd	0.398100	1.3	keep	increase encoder depth to 6
working	0.000000	0.0	crash	make predictor twice as wide (OOM)
2d3e4f5	0.421000	1.2	discard	switch gaussian reg weight to 1.0
```

**Do NOT commit results.tsv.** Leave it untracked by git.

## The experiment loop

LOOP FOREVER:

1. Check git state: confirm which commit you are on and that `train.py` has no unexpected changes.
2. Pick one concrete idea and implement it in `train.py`.
3. `git commit` the change with a short message.
4. Remove any stale `checkpoint.pt` so you do not accidentally evaluate an old checkpoint.
5. Run training: `uv run --no-sync train.py > run.log 2>&1`
6. If training crashes, inspect: `tail -n 50 run.log`
7. Run evaluation: `uv run --no-sync eval.py --device <device> > eval.log 2>&1`
8. Extract the metric: `grep "pred_error" eval.log`
9. Query peak GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
10. Record the result in `results.tsv`.
11. If `pred_error` improved (lower): keep the commit — the branch advances.
12. If `pred_error` did not improve: `git reset --hard HEAD~1` to revert the commit and restore the last kept version.

**Timeout**: Each training run should complete in under 10 minutes (wall clock, including startup). If a run exceeds 10 minutes, kill it with `kill <pid>` and treat it as a failure — log `crash`, revert, and move on.

**Crashes**: Use your judgment. If it's something easy to fix (typo, missing import), fix it and re-run. If the idea is fundamentally broken (OOM, bad architecture), log `crash` as the status, `git reset --hard HEAD~1`, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer, and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Guardrails

- Only compare runs using the same dataset and evaluation path.
- For real comparisons, use full training runs. Short runs are only for smoke testing after risky refactors.
- Do not overwrite unrelated dirty files.
- If you see unexpected edits to `train.py` that you did not make, pause and ask before continuing.
- If a run does not produce a fresh `checkpoint.pt`, treat it as a failed run.
