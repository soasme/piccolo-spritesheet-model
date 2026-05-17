import torch
import pytest
from pathlib import Path
from PIL import Image


def _make_checkpoint(tmp_path):
    from train import SpriteEncoder, SpritePredictor
    encoder, predictor = SpriteEncoder(), SpritePredictor()
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}, ckpt)
    return ckpt


def _make_frames(tmp_path, name="test", rows=1, cols=3):
    d = tmp_path / name
    d.mkdir()
    for r in range(rows):
        for c in range(cols):
            Image.new("RGB", (32, 32), color=((r * 50) % 256, (c * 40) % 256, 100)).save(
                d / f"{name}_r{r:02d}_c{c:02d}.png"
            )
    return tmp_path


def test_compute_pred_error_returns_float(tmp_path):
    from eval import compute_pred_error

    ckpt = _make_checkpoint(tmp_path)
    frames_dir = _make_frames(tmp_path)
    error = compute_pred_error(ckpt, frames_dir, device="cpu")
    assert isinstance(error, float)


def test_compute_pred_error_in_range(tmp_path):
    from eval import compute_pred_error

    ckpt = _make_checkpoint(tmp_path)
    frames_dir = _make_frames(tmp_path)
    error = compute_pred_error(ckpt, frames_dir, device="cpu")
    assert 0.0 <= error <= 2.0, f"cosine distance out of [0,2]: {error}"
