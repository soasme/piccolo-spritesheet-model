from pathlib import Path
from collections import defaultdict
import json
import time
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
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
TRAINING_BUDGET = 600   # wall-clock seconds of training (excluding startup)
MAX_STEPS = 10_000      # used when time budget is disabled
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
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


# ── Decoder ────────────────────────────────────────────────────────────────
class SpriteDecoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, out_channels: int = 4):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16→32
            nn.GELU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.proj(z).reshape(z.shape[0], 512, 4, 4))


# ── Dataset ────────────────────────────────────────────────────────────────
FRAME_SIZE = 64  # all frames resized to this square; must be divisible by PATCH_SIZE

_frame_transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
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
def train(time_budget: int | None = TRAINING_BUDGET, batch_size: int = BATCH_SIZE, device: str = "auto") -> None:
    dev_str = _get_device(device)
    dev = torch.device(dev_str)
    is_cuda = dev_str == "cuda"

    dataset = SpriteFrameDataset(FRAMES_DIR)
    if len(dataset) == 0:
        raise RuntimeError(f"No frame pairs found in {FRAMES_DIR}. Run prepare.py first.")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        drop_last=True,
    )
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    decoder = SpriteDecoder().to(dev)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    step, best_loss = 0, float("inf")
    data_iter = iter(loader)
    t_start = time.time()
    budget_str = f"{time_budget}s" if time_budget is not None else f"{MAX_STEPS} steps"
    print(f"Training on {len(dataset)} frame pairs  device={dev_str}  budget={budget_str}")

    while (time_budget is not None and time.time() - t_start < time_budget) or \
          (time_budget is None and step < MAX_STEPS):
        try:
            frame_t, frame_t1, _texts = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frame_t, frame_t1, _texts = next(data_iter)

        frame_t, frame_t1 = frame_t.to(dev), frame_t1.to(dev)
        frame_t1_gen = decoder(predictor(encoder(frame_t)))
        loss = F.l1_loss(frame_t1_gen, frame_t1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            print(f"step {step:>6d}  loss={loss.item():.4f}  t={elapsed:.0f}s")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "predictor": predictor.state_dict(),
                        "decoder": decoder.state_dict(),
                    },
                    CHECKPOINT_PATH,
                )
        step += 1

    training_seconds = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated(dev) / 1024**2 if is_cuda else 0.0
    print(f"---")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--time-budget", type=int, default=TRAINING_BUDGET,
                   help="wall-clock training seconds (default: 300)")
    p.add_argument("--no-time-budget", dest="time_budget", action="store_const", const=None,
                   help=f"disable time budget; train for {MAX_STEPS} steps instead")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", type=str, default="auto",
                   help="cuda | mps | cpu | auto (default: auto-detect)")
    args = p.parse_args()
    train(time_budget=args.time_budget, batch_size=args.batch_size, device=args.device)
