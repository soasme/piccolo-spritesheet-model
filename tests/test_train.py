import torch
import torch.nn.functional as F
import pytest
from pathlib import Path
from PIL import Image


def test_sprite_encoder_output_shape():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    z = encoder(torch.randn(4, 4, 32, 32))
    assert z.shape == (4, 128), f"expected (4, 128), got {z.shape}"


def test_sprite_encoder_batch_size_invariant():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    for B in [1, 8, 16]:
        z = encoder(torch.randn(B, 4, 32, 32))
        assert z.shape == (B, 128)


def test_sprite_encoder_gradients_flow():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    loss = encoder(torch.randn(2, 4, 32, 32)).mean()
    loss.backward()
    assert any(p.grad is not None for p in encoder.parameters())


def test_sprite_predictor_output_shape():
    from train import SpritePredictor, LATENT_DIM

    pred = SpritePredictor()
    z_pred = pred(torch.randn(4, LATENT_DIM))
    assert z_pred.shape == (4, LATENT_DIM)


def test_gaussian_reg_is_scalar_and_nonneg():
    from train import gaussian_reg

    loss = gaussian_reg(torch.randn(32, 128))
    assert loss.shape == (), f"expected scalar, got {loss.shape}"
    assert loss.item() >= 0


def test_gaussian_reg_near_zero_for_standard_normal():
    from train import gaussian_reg

    loss = gaussian_reg(torch.randn(2048, 128))
    assert loss.item() < 0.1, f"reg on N(0,1) should be small, got {loss.item():.4f}"


def _make_frame_dir(tmp_path, name, rows, cols):
    d = tmp_path / name
    d.mkdir()
    for r in range(rows):
        for c in range(cols):
            Image.new("RGBA", (32, 32), color=((r * 50) % 256, (c * 40) % 256, 100, 255)).save(
                d / f"{name}_r{r:02d}_c{c:02d}.png"
            )
    return tmp_path


def test_dataset_pair_count(tmp_path):
    from train import SpriteFrameDataset

    _make_frame_dir(tmp_path, "hero", rows=2, cols=4)
    ds = SpriteFrameDataset(tmp_path)
    assert len(ds) == 6  # 2 rows × 3 consecutive pairs per row


def test_dataset_item_shapes(tmp_path):
    from train import SpriteFrameDataset

    _make_frame_dir(tmp_path, "hero", rows=1, cols=3)
    ds = SpriteFrameDataset(tmp_path)
    frame_t, frame_t1, text = ds[0]
    assert frame_t.shape == (4, 32, 32)
    assert frame_t1.shape == (4, 32, 32)
    assert isinstance(text, str)


def test_dataset_text_from_metadata(tmp_path):
    import json
    from train import SpriteFrameDataset

    _make_frame_dir(tmp_path, "hero", rows=1, cols=3)
    meta = tmp_path / "hero" / "metadata.json"
    meta.write_text(json.dumps({"text": "3-frame sprite animation of: hero", "n_frames": 3}))
    ds = SpriteFrameDataset(tmp_path)
    _, _, text = ds[0]
    assert text == "3-frame sprite animation of: hero"


def test_dataset_text_empty_without_metadata(tmp_path):
    from train import SpriteFrameDataset

    _make_frame_dir(tmp_path, "hero", rows=1, cols=3)
    ds = SpriteFrameDataset(tmp_path)
    _, _, text = ds[0]
    assert text == ""


def test_dataset_empty_dir(tmp_path):
    from train import SpriteFrameDataset

    ds = SpriteFrameDataset(tmp_path)
    assert len(ds) == 0


def test_jepa_training_step_produces_finite_loss():
    from train import SpriteEncoder, SpritePredictor, gaussian_reg, LAMBDA_REG

    encoder = SpriteEncoder()
    predictor = SpritePredictor()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4
    )
    frame_t = torch.randn(8, 4, 32, 32)
    frame_t1 = torch.randn(8, 4, 32, 32)

    z_t = encoder(frame_t)
    z_t1_pred = predictor(z_t)
    with torch.no_grad():
        z_t1 = encoder(frame_t1)

    loss = F.mse_loss(z_t1_pred, z_t1) + LAMBDA_REG * gaussian_reg(z_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.item() > 0
