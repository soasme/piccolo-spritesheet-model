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
