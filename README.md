# piccolo-spritesheet

A minimal JEPA latent world model for pixel-art sprite animation. Give it sprite sheets, it learns to predict the next frame in latent space — no decoder, no diffusion, no manual cleanup.

## How it works

The repo is deliberately kept small and only has three files that matter:

- **`prepare.py`** — downloads sprite sheets from OpenGameArt and HuggingFace, slices them into frame pairs, and builds the training dataset. Not modified.
- **`train.py`** — ViT encoder, transformer predictor, and JEPA training loop. Everything is fair game: architecture, loss, optimizer, hyperparameters. **This file is edited and iterated on.**
- **`eval.py`** — measures identity consistency, palette consistency, and temporal smoothness. Not modified.

The metric is **`pred_error`** (latent cosine distance between predicted and actual next frame) — lower is better, resolution-independent.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download sprites and slice into frames (one-time, ~5 min)
uv run prepare.py

# 4. Train
uv run train.py

# 5. Evaluate
uv run eval.py
```

## Project structure

```
prepare.py     — data download + frame slicing  (do not modify)
train.py       — encoder, predictor, loop        (modify this)
eval.py        — evaluation metrics              (do not modify)
pyproject.toml — dependencies
data/          — downloaded sprites + sliced frames
```

## Design choices

- **JEPA, not reconstruction.** The model predicts in latent space, not pixels. This avoids blur collapse, palette drift, and subpixel noise that plague pixel-reconstruction objectives.
- **Sprite-native metric.** `pred_error` measures how well the model predicts motion dynamics, not perceptual realism. Identity and palette stability are what matter for game-usable sprites.
- **Single file to modify.** The agent (or you) only touches `train.py`. Diffs stay reviewable, scope stays manageable.
- **Small models first.** 32×32 sprite target. Real-time inference preferred over scale.

## Research notes

See [`docs/IDEA.md`](docs/IDEA.md) for the full research vision, architectural ideas, and long-term direction.

## License

MIT
