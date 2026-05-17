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
