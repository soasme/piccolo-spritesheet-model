from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import einops

# ── Constants ──────────────────────────────────────────────────────────────
FRAME_SIZE = 32
LATENT_DIM = 128
HIDDEN_DIM = 128
PREDICTOR_HIDDEN = 256
PATCH_SIZE = 4
DEPTH = 4
HEADS = 4
LAMBDA_REG = 0.1
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_STEPS = 10_000
LOG_EVERY = 100
CHECKPOINT_PATH = Path("checkpoint.pt")
DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "frames"


# ── Encoder ────────────────────────────────────────────────────────────────
class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(self.proj(x), "b d h w -> b (h w) d")


class SpriteEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = FRAME_SIZE,
        patch_size: int = PATCH_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        depth: int = DEPTH,
        heads: int = HEADS,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = _PatchEmbed(img_size, patch_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.head(self.norm(self.transformer(x)[:, 0]))
