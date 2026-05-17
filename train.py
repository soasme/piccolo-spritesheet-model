from pathlib import Path
from collections import defaultdict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import einops

# ── Constants ──────────────────────────────────────────────────────────────
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


# ── Device ─────────────────────────────────────────────────────────────────
def _get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Positional embedding ────────────────────────────────────────────────────
def _build_2d_sinusoidal(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """2D sinusoidal positional embedding for an h×w patch grid. Returns (1, h*w, dim)."""
    assert dim % 4 == 0
    half = dim // 2
    freqs = torch.arange(half // 2, device=device).float()
    freqs = 1.0 / (10000 ** (freqs / (half // 2)))
    y_enc = torch.outer(torch.arange(h, device=device).float(), freqs)
    x_enc = torch.outer(torch.arange(w, device=device).float(), freqs)
    y_enc = torch.cat([y_enc.sin(), y_enc.cos()], dim=-1)  # (h, half)
    x_enc = torch.cat([x_enc.sin(), x_enc.cos()], dim=-1)  # (w, half)
    grid = torch.cat([
        y_enc[:, None, :].expand(h, w, half),
        x_enc[None, :, :].expand(h, w, half),
    ], dim=-1)                                              # (h, w, dim)
    return grid.reshape(1, h * w, dim)                     # (1, h*w, dim)


# ── Encoder ────────────────────────────────────────────────────────────────
class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_dim: int, in_channels: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(self.proj(x), "b d h w -> b (h w) d")


class SpriteEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = PATCH_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        depth: int = DEPTH,
        heads: int = HEADS,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = _PatchEmbed(patch_size, hidden_dim, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        tokens = self.patch_embed(x)
        tokens = tokens + _build_2d_sinusoidal(h_p, w_p, tokens.shape[-1], x.device)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        return self.head(self.norm(self.transformer(x)[:, 0]))


# ── Predictor ──────────────────────────────────────────────────────────────
class SpritePredictor(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = PREDICTOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Loss ───────────────────────────────────────────────────────────────────
def gaussian_reg(z: torch.Tensor) -> torch.Tensor:
    """Push latents toward N(0,1): mean→0, std→1 per dimension."""
    mean_loss = z.mean(0).pow(2).mean()
    var_loss = (1 - z.std(0).clamp(min=1e-6)).pow(2).mean()
    return mean_loss + var_loss


# ── Dataset ────────────────────────────────────────────────────────────────
_frame_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
])


def _pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Pad variable-size frames to the largest H×W in the batch."""
    frames_t, frames_t1, texts = zip(*batch)
    all_f = frames_t + frames_t1
    max_h = max(f.shape[1] for f in all_f)
    max_w = max(f.shape[2] for f in all_f)

    def pad(frames: tuple) -> torch.Tensor:
        out = []
        for f in frames:
            c, h, w = f.shape
            p = torch.zeros(c, max_h, max_w, dtype=f.dtype)
            p[:, :h, :w] = f
            out.append(p)
        return torch.stack(out)

    return pad(frames_t), pad(frames_t1), list(texts)


class SpriteFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames_root: Path):
        self.pairs: list[tuple[Path, Path, str]] = []
        groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for p in frames_root.rglob("*_r[0-9][0-9]_c[0-9][0-9].png"):
            base, col_str = p.stem.rsplit("_c", 1)
            groups[base].append((int(col_str), p))
        for frames in groups.values():
            frames.sort()
            meta_path = frames[0][1].parent / "metadata.json"
            text = ""
            if meta_path.exists():
                text = json.loads(meta_path.read_text(encoding="utf-8")).get("text", "")
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i][1], frames[i + 1][1], text))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        p_t, p_t1, text = self.pairs[idx]
        return (
            _frame_transform(Image.open(p_t).convert("RGBA")),
            _frame_transform(Image.open(p_t1).convert("RGBA")),
            text,
        )


# ── Training loop ──────────────────────────────────────────────────────────
def train(steps: int = MAX_STEPS, batch_size: int = BATCH_SIZE, device: str = "auto") -> None:
    dev_str = _get_device(device)
    dev = torch.device(dev_str)
    is_cuda = dev_str == "cuda"

    dataset = SpriteFrameDataset(FRAMES_DIR)
    if len(dataset) == 0:
        raise RuntimeError(f"No frame pairs found in {FRAMES_DIR}. Run prepare.py first.")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_pad_collate,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        drop_last=True,
    )
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    step, best_loss = 0, float("inf")
    data_iter = iter(loader)
    print(f"Training on {len(dataset)} frame pairs  device={dev_str}")

    while step < steps:
        try:
            frame_t, frame_t1, _texts = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frame_t, frame_t1, _texts = next(data_iter)

        frame_t, frame_t1 = frame_t.to(dev), frame_t1.to(dev)
        z_t = encoder(frame_t)
        z_t1_pred = predictor(z_t)
        with torch.no_grad():
            z_t1 = encoder(frame_t1)

        loss = F.mse_loss(z_t1_pred, z_t1) + LAMBDA_REG * gaussian_reg(z_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            pred_error = (1 - F.cosine_similarity(z_t1_pred.detach(), z_t1).mean()).item()
            print(f"step {step:>6d}  loss={loss.item():.4f}  pred_error={pred_error:.4f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()},
                    CHECKPOINT_PATH,
                )
        step += 1

    print(f"Done. Best loss: {best_loss:.4f}  checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", type=str, default="auto",
                   help="cuda | mps | cpu | auto (default: auto-detect)")
    args = p.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, device=args.device)
