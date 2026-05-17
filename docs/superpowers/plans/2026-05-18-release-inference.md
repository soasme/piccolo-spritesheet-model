# Release + Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `release.py` to push the trained checkpoint to HuggingFace Hub and `inference.py` to predict the next frame (or a sequence) via nearest-neighbor retrieval in latent space.

**Architecture:** `release.py` validates and uploads `checkpoint.pt` plus a model card to `soasme/piccolo-spritesheet-model` using `huggingface_hub`. `inference.py` loads the model, encodes all dataset frames into a latent index, then for each prediction step encodes the input frame → predicts the next latent → finds the closest real frame by cosine similarity.

**Tech Stack:** PyTorch, Pillow, huggingface_hub; imports `SpriteEncoder`, `SpritePredictor`, `_frame_transform`, `_get_device`, `CHECKPOINT_PATH`, `FRAMES_DIR` from `train.py`.

**Working directory:** All commands run from inside `.worktrees/feat-inference/`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `huggingface_hub>=0.20.0` to dependencies |
| `release.py` | Create | Validate checkpoint, build model card, upload to HF Hub |
| `inference.py` | Create | Load model, build latent index, NN retrieval, CLI |
| `tests/test_release.py` | Create | Unit tests for validate_checkpoint, build_model_card |
| `tests/test_inference.py` | Create | Unit tests for load_model, build_latent_index, find_nearest, encode_frame, predict_next |

---

### Task 1: Add huggingface_hub dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, add `huggingface_hub>=0.20.0` to the `dependencies` list:

```toml
[project]
name = "piccolo-spritesheet"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "Pillow>=10.0.0",
    "numpy>=1.26.0",
    "einops>=0.7.0",
    "requests>=2.31.0",
    "datasets>=2.14.0",
    "huggingface_hub>=0.20.0",
]
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from huggingface_hub import HfApi; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add huggingface_hub dependency"
```

---

### Task 2: Write failing tests for release.py

**Files:**
- Create: `tests/test_release.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_release.py`:

```python
import pytest
from pathlib import Path
import torch
import tempfile
from train import SpriteEncoder, SpritePredictor


def _make_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "encoder": SpriteEncoder().state_dict(),
            "predictor": SpritePredictor().state_dict(),
        },
        ckpt,
    )
    return ckpt


def test_validate_checkpoint_missing():
    from release import validate_checkpoint
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_checkpoint(Path("nonexistent_abc.pt"))


def test_validate_checkpoint_valid(tmp_path):
    from release import validate_checkpoint
    ckpt = _make_checkpoint(tmp_path)
    state = validate_checkpoint(ckpt)
    assert "encoder" in state
    assert "predictor" in state


def test_validate_checkpoint_missing_keys(tmp_path):
    from release import validate_checkpoint
    bad = tmp_path / "bad.pt"
    torch.save({"encoder": {}}, bad)
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint(bad)


def test_build_model_card_contains_required_sections():
    from release import build_model_card
    card = build_model_card()
    assert "piccolo-spritesheet-model" in card
    assert "JEPA" in card
    assert "64" in card
    assert "soasme/piccolo-spritesheet-model" in card
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_release.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `release` module not found.

---

### Task 3: Implement release.py

**Files:**
- Create: `release.py`

- [ ] **Step 1: Create release.py**

```python
from pathlib import Path
import torch
from huggingface_hub import HfApi

REPO_ID = "soasme/piccolo-spritesheet-model"


def validate_checkpoint(checkpoint: Path) -> dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "encoder" not in state or "predictor" not in state:
        raise ValueError(f"Checkpoint missing 'encoder' or 'predictor' keys: {checkpoint}")
    return state


def build_model_card() -> str:
    from train import LATENT_DIM, HIDDEN_DIM, PATCH_SIZE, DEPTH, HEADS
    return f"""---
language: en
tags:
  - sprite
  - pixel-art
  - world-model
  - jepa
---

# piccolo-spritesheet-model

JEPA-style latent predictor for pixel-art sprite animation.

## Architecture

- Encoder: ViT-style patch encoder (`patch_size={PATCH_SIZE}`, `hidden_dim={HIDDEN_DIM}`, `depth={DEPTH}`, `heads={HEADS}`)
- Predictor: 3-layer MLP operating in latent space
- Latent dim: {LATENT_DIM}
- Input: 64×64 RGBA frames

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from train import SpriteEncoder, SpritePredictor

ckpt = torch.load(
    hf_hub_download("soasme/piccolo-spritesheet-model", "checkpoint.pt"),
    weights_only=True,
)
encoder = SpriteEncoder()
predictor = SpritePredictor()
encoder.load_state_dict(ckpt["encoder"])
predictor.load_state_dict(ckpt["predictor"])
encoder.eval()
predictor.eval()
```

See `inference.py` in the repo for next-frame prediction via nearest-neighbor retrieval.
"""


def release(checkpoint: Path = Path("checkpoint.pt"), repo: str = REPO_ID) -> None:
    validate_checkpoint(checkpoint)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="checkpoint.pt",
        repo_id=repo,
        repo_type="model",
    )
    card = build_model_card()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )
    print(f"Released to https://huggingface.co/{repo}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    p.add_argument("--repo", type=str, default=REPO_ID)
    args = p.parse_args()
    release(checkpoint=args.checkpoint, repo=args.repo)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_release.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add release.py tests/test_release.py
git commit -m "feat: add release.py with HuggingFace upload and model card"
```

---

### Task 4: Write failing tests for inference.py

**Files:**
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inference.py`:

```python
import pytest
from pathlib import Path
import torch
from PIL import Image
from train import SpriteEncoder, SpritePredictor, LATENT_DIM


@pytest.fixture
def checkpoint_path(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "encoder": SpriteEncoder().state_dict(),
            "predictor": SpritePredictor().state_dict(),
        },
        ckpt,
    )
    return ckpt


@pytest.fixture
def frames_dir(tmp_path):
    d = tmp_path / "frames" / "hero"
    d.mkdir(parents=True)
    for i in range(4):
        img = Image.new("RGBA", (64, 64), color=(i * 60, 100, 150, 255))
        img.save(d / f"hero_r00_c{i:02d}.png")
    return tmp_path / "frames"


def test_load_model_returns_encoder_and_predictor(checkpoint_path):
    from inference import load_model
    encoder, predictor = load_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(encoder, SpriteEncoder)
    assert isinstance(predictor, SpritePredictor)


def test_load_model_sets_eval_mode(checkpoint_path):
    from inference import load_model
    encoder, predictor = load_model(checkpoint_path, torch.device("cpu"))
    assert not encoder.training
    assert not predictor.training


def test_build_latent_index_shape(checkpoint_path, frames_dir):
    from inference import load_model, build_latent_index
    encoder, _ = load_model(checkpoint_path, torch.device("cpu"))
    stack, paths = build_latent_index(encoder, frames_dir, torch.device("cpu"))
    assert stack.shape == (4, LATENT_DIM)
    assert len(paths) == 4


def test_build_latent_index_empty_dir(checkpoint_path, tmp_path):
    from inference import load_model, build_latent_index
    encoder, _ = load_model(checkpoint_path, torch.device("cpu"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No PNG"):
        build_latent_index(encoder, empty, torch.device("cpu"))


def test_find_nearest_returns_path_from_index(checkpoint_path, frames_dir):
    from inference import load_model, build_latent_index, find_nearest
    encoder, _ = load_model(checkpoint_path, torch.device("cpu"))
    stack, paths = build_latent_index(encoder, frames_dir, torch.device("cpu"))
    z_pred = torch.randn(LATENT_DIM)
    result = find_nearest(z_pred, stack, paths)
    assert result in paths


def test_find_nearest_identical_latent_returns_same_frame(checkpoint_path, frames_dir):
    from inference import load_model, build_latent_index, find_nearest
    encoder, _ = load_model(checkpoint_path, torch.device("cpu"))
    stack, paths = build_latent_index(encoder, frames_dir, torch.device("cpu"))
    # Query with first indexed latent — must return that same frame
    result = find_nearest(stack[0], stack, paths)
    assert result == paths[0]


def test_encode_frame_returns_latent_vector(checkpoint_path, frames_dir):
    from inference import load_model, encode_frame
    encoder, _ = load_model(checkpoint_path, torch.device("cpu"))
    paths = sorted(frames_dir.rglob("*.png"))
    z = encode_frame(paths[0], encoder, torch.device("cpu"))
    assert z.shape == (LATENT_DIM,)


def test_predict_next_returns_path_in_index(checkpoint_path, frames_dir):
    from inference import load_model, build_latent_index, encode_frame, predict_next
    encoder, predictor = load_model(checkpoint_path, torch.device("cpu"))
    stack, paths = build_latent_index(encoder, frames_dir, torch.device("cpu"))
    z_t = encode_frame(paths[0], encoder, torch.device("cpu"))
    _, nearest = predict_next(z_t, predictor, stack, paths)
    assert nearest in paths
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: `ImportError` — `inference` module not found.

---

### Task 5: Implement inference.py core functions

**Files:**
- Create: `inference.py`

- [ ] **Step 1: Create inference.py with core functions and CLI**

```python
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from PIL import Image
from train import (
    SpriteEncoder,
    SpritePredictor,
    _frame_transform,
    CHECKPOINT_PATH,
    FRAMES_DIR,
    _get_device,
)


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[SpriteEncoder, SpritePredictor]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder = SpriteEncoder().to(device)
    predictor = SpritePredictor().to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor


def build_latent_index(
    encoder: SpriteEncoder, frames_dir: Path, device: torch.device
) -> tuple[torch.Tensor, list[Path]]:
    paths = sorted(frames_dir.rglob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    latents = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGBA")
            x = _frame_transform(img).unsqueeze(0).to(device)
            latents.append(encoder(x).squeeze(0))
    return torch.stack(latents), paths


def find_nearest(
    z_pred: torch.Tensor, latent_stack: torch.Tensor, frame_paths: list[Path]
) -> Path:
    sims = F.cosine_similarity(z_pred.unsqueeze(0), latent_stack, dim=1)
    return frame_paths[sims.argmax().item()]


def encode_frame(
    path: Path, encoder: SpriteEncoder, device: torch.device
) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    x = _frame_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(x).squeeze(0)


def predict_next(
    z_t: torch.Tensor,
    predictor: SpritePredictor,
    latent_stack: torch.Tensor,
    frame_paths: list[Path],
) -> tuple[torch.Tensor, Path]:
    with torch.no_grad():
        z_t1_pred = predictor(z_t.unsqueeze(0)).squeeze(0)
    nearest = find_nearest(z_t1_pred, latent_stack, frame_paths)
    return z_t1_pred, nearest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sprite frame prediction via NN retrieval")
    p.add_argument("--input", type=Path, required=True, help="Input frame (PNG)")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR,
                   help="Dataset frames directory for building latent index")
    p.add_argument("--output", type=Path, default=Path("predicted.png"),
                   help="Output path (single-frame mode)")
    p.add_argument("--frame-count", type=int, default=None,
                   help="Predict this many frames (sequence mode)")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="Output directory (sequence mode)")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    dev = torch.device(_get_device(args.device))
    encoder, predictor = load_model(args.checkpoint, dev)
    print(f"Building latent index from {args.frames_dir} ...")
    latent_stack, frame_paths = build_latent_index(encoder, args.frames_dir, dev)
    print(f"  Indexed {len(frame_paths)} frames.")

    if args.frame_count is None:
        # Single-frame mode
        z_t = encode_frame(args.input, encoder, dev)
        _, nearest = predict_next(z_t, predictor, latent_stack, frame_paths)
        shutil.copy2(nearest, args.output)
        print(f"Predicted next frame → {args.output}  (source: {nearest})")
    else:
        # Sequence mode
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input, args.output_dir / "frame_000.png")
        print(f"frame_000.png  (input)")
        current_path = args.input
        for i in range(1, args.frame_count + 1):
            z_t = encode_frame(current_path, encoder, dev)
            _, nearest = predict_next(z_t, predictor, latent_stack, frame_paths)
            out_name = f"frame_{i:03d}.png"
            shutil.copy2(nearest, args.output_dir / out_name)
            print(f"{out_name}  (source: {nearest})")
            current_path = nearest
        print(f"Sequence saved to {args.output_dir}/")
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add inference.py tests/test_inference.py
git commit -m "feat: add inference.py with latent-NN frame prediction"
```

---

### Task 6: Smoke-test the full CLI

**Files:** none (no changes, just verification)

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS (including pre-existing `test_eval.py`, `test_prepare.py`, `test_train.py`).

- [ ] **Step 2: Verify inference.py --help works**

```bash
uv run python inference.py --help
```

Expected: usage text listing `--input`, `--checkpoint`, `--frames-dir`, `--output`, `--frame-count`, `--output-dir`, `--device`.

- [ ] **Step 3: Verify release.py --help works**

```bash
uv run python release.py --help
```

Expected: usage text listing `--checkpoint`, `--repo`.

- [ ] **Step 4: Commit if any fixes were needed; otherwise no commit**

```bash
git status
```

If clean, nothing to do. If there were smoke-test fixes, stage and commit them.
