import torch
import torch.nn.functional as F
import pytest
from pathlib import Path
from PIL import Image


def test_sprite_encoder_output_shape():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    z = encoder(torch.randn(4, 3, 32, 32))
    assert z.shape == (4, 128), f"expected (4, 128), got {z.shape}"


def test_sprite_encoder_batch_size_invariant():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    for B in [1, 8, 16]:
        z = encoder(torch.randn(B, 3, 32, 32))
        assert z.shape == (B, 128)


def test_sprite_encoder_gradients_flow():
    from train import SpriteEncoder

    encoder = SpriteEncoder()
    loss = encoder(torch.randn(2, 3, 32, 32)).mean()
    loss.backward()
    assert any(p.grad is not None for p in encoder.parameters())
