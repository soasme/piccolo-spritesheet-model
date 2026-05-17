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
