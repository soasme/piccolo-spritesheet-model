# piccolo-spritesheet autoresearch

This repo is set up for autonomous experiments on a small JEPA sprite model.

The job is simple: improve `train.py` to get the lowest possible `pred_error` from `eval.py`.

## Setup

Before starting a run, work with the user to:

1. Agree on a run tag based on the date, for example `may17` or `2026-05-17`.
2. Inspect `git status --short` before touching anything.
3. If `train.py` already has uncommitted changes that you did not make, stop and ask the user how to proceed. Do not overwrite user work.
4. Create a dedicated branch such as `autoresearch/<tag>` from the current baseline commit.
5. Read these files for full context:
   - `README.md` - repository context and constraints
   - `prepare.py` - data download and frame slicing; do not modify
   - `train.py` - the only file you optimize
   - `eval.py` - evaluation entrypoint and source-of-truth metric
6. Verify dependencies are installed with `uv sync` if needed.
7. Verify data exists under `data/frames/`. If not, run `uv run prepare.py` first. If network access is unavailable, ask the human to prepare the data.
8. Create `results.tsv` with this header row:

```tsv
commit	pred_error	status	description
```

9. Pick an evaluation device:
   - `cuda` if CUDA is available
   - `mps` on Apple Silicon if CUDA is not available
   - `cpu` otherwise

Once setup is confirmed, start the experimentation loop.

## Experimentation

Each experiment is a full training run using the current repo defaults:

```bash
uv run train.py
```

Then evaluate the checkpoint:

```bash
uv run eval.py --device <device>
```

### What you CAN do

- Modify `train.py`
- Change model architecture, loss weighting, optimizer, scheduler, batch size, latent size, predictor design, logging, checkpointing, and training loop details inside `train.py`

### What you CANNOT do

- Modify `prepare.py`
- Modify `eval.py`
- Change the dataset contents by hand
- Add new dependencies
- Change the metric definition

### Goal

Minimize `pred_error`. Lower is better.

Training loss is only a proxy. `eval.py` is the source of truth.

### Simplicity criterion

Prefer changes that are easy to reason about and easy to keep. A tiny improvement is not worth a messy hack. A simple change that gives the same or better result is a win.

### First run

The first run should always be the baseline with no code changes.

## Output format

Training prints progress lines like:

```text
step      0  loss=0.1234  pred_error=0.5678
```

Evaluation prints the metric in this form:

```text
pred_error (mean cosine distance): 0.1234
```

Use the evaluation output, not the training log, to judge experiments.

## Logging results

After every experiment, append one row to `results.tsv`.

Columns:

1. `commit` - short git hash for kept runs; use `working` for uncommitted crash/debug runs
2. `pred_error` - numeric metric from `eval.py`; use `0.0000` for crashes
3. `status` - `keep`, `discard`, or `crash`
4. `description` - short description of the idea tested

Example:

```tsv
commit	pred_error	status	description
4f2c1ab	0.4123	keep	baseline
9a8b7cd	0.3981	keep	increase encoder depth to 6
working	0.0000	crash	make predictor twice as wide
2d3e4f5	0.4210	discard	switch gaussian reg weight to 1.0
```

## Experiment loop

Repeat until the user interrupts you:

1. Check git state and confirm you understand which changes are yours.
2. Pick one concrete idea and implement it in `train.py`.
3. Before running, remove or replace any stale `checkpoint.pt` so you do not accidentally evaluate an old checkpoint.
4. Run training with output redirected:

```bash
uv run train.py > run.log 2>&1
```

5. If training crashes, inspect:

```bash
tail -n 50 run.log
```

6. If the crash is a small bug, fix it and rerun. If the idea is fundamentally bad, record a `crash` row and move on.
7. After a successful training run, evaluate with output redirected:

```bash
uv run eval.py --device <device> > eval.log 2>&1
```

8. Extract the metric:

```bash
grep "pred_error" eval.log
```

9. Record the result in `results.tsv`.
10. If `pred_error` improved, keep the change and commit it with a short message.
11. If `pred_error` did not improve, discard only your `train.py` experiment and restore the last kept version without touching unrelated user changes.
12. Repeat.

## Guardrails

- Only compare runs using the same dataset and evaluation path.
- For real comparisons, use full training runs. Short runs are only for smoke testing after risky refactors.
- Do not overwrite unrelated dirty files.
- If you see unexpected edits to `train.py`, pause and ask before continuing.
- If a run does not produce a fresh `checkpoint.pt`, treat it as a failed run.
