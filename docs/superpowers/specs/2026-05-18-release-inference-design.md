# Design: release.py + inference.py

**Date:** 2026-05-18  
**Status:** Approved

---

## Context

The project trains a JEPA-style latent predictor for pixel-art sprite animation.

- `SpriteEncoder` encodes a 64×64 RGBA frame → 128-dim latent vector
- `SpritePredictor` maps latent_t → latent_t+1
- No decoder exists; the model operates entirely in latent space

Checkpoint is saved to `checkpoint.pt`.

---

## release.py

Uploads the trained checkpoint to `soasme/piccolo-spritesheet-model` on HuggingFace Hub.

### CLI

```
python release.py [--checkpoint checkpoint.pt] [--repo soasme/piccolo-spritesheet-model]
```

### Steps

1. Verify checkpoint exists and is loadable (`torch.load`, `weights_only=True`)
2. Authenticate via `HfApi` — reads `HF_TOKEN` from env or `~/.cache/huggingface`
3. Create repo if it doesn't exist (`repo_type="model"`, `exist_ok=True`)
4. Upload `checkpoint.pt` via `upload_file`
5. Generate and upload a minimal `README.md` model card (architecture constants, training notes)

### Dependencies

Add `huggingface_hub` explicitly to `pyproject.toml` (already transitively available via `datasets`).

---

## inference.py

Loads the trained model and predicts the next frame (or a sequence of frames) using nearest-neighbor retrieval in latent space.

Since the model has no pixel decoder, "predicting the next frame" means:
1. Encode the input frame → z_t
2. Predict → z_t+1 in latent space
3. Find the real frame from the dataset whose latent is closest (cosine similarity)
4. Output that frame as the result

### CLI

```
# Predict next frame only
python inference.py --input frame.png --frames-dir data/frames --output out.png

# Predict N frames in sequence
python inference.py --input frame.png --frames-dir data/frames --frame-count 8 --output-dir out/
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to input frame (PNG) |
| `--checkpoint` | `checkpoint.pt` | Path to model checkpoint |
| `--frames-dir` | `data/frames` | Dataset frames directory used to build latent index |
| `--output` | `predicted.png` | Output path for single-frame mode |
| `--frame-count` | — | If given, predict this many frames and write to `--output-dir` |
| `--output-dir` | `output/` | Directory for multi-frame mode |
| `--device` | `auto` | cuda / mps / cpu / auto |

### Implementation

**Build latent index:** Walk all `*.png` files in `--frames-dir`, encode each with `SpriteEncoder`, store as parallel lists of `(latent, path)`. Built once per invocation.

**Single-frame prediction:**
- Load and transform `--input` with `_frame_transform` from `train.py`
- Encode → z_t, predict → z_t1_pred
- Cosine similarity against all indexed latents → argmax
- Save matched frame to `--output`

**Sequence prediction (`--frame-count N`):**
- Save input frame as `output/frame_000.png`
- Loop N times: retrieve nearest frame, save as `frame_NNN.png`, encode it as new input
- Produces `N+1` files total (input + N predicted)

**Reuse from `train.py`:** Import `SpriteEncoder`, `SpritePredictor`, `_frame_transform`, `CHECKPOINT_PATH`, `FRAMES_DIR`, `_get_device` directly — no duplication.

---

## Out of Scope

- Pixel decoder / reconstruction
- Loading model from HuggingFace Hub during inference
- Pre-encoded latent index bundled in the release
