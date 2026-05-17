# Piccolo-Spritesheet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-file JEPA world model that downloads pixel-art sprites from OpenGameArt, slices them into consecutive frame pairs, and trains a ViT encoder + MLP predictor to predict the next animation frame in latent space.

**Architecture:** `prepare.py` downloads sprite sheets and slices them into 32×32 frames named `{sprite}_r{row:02d}_c{col:02d}.png`. `train.py` defines a ViT encoder (128-dim latent) and MLP predictor trained with JEPA loss: MSE against stop-gradient target latents + Gaussian regularization to prevent collapse. `eval.py` loads a checkpoint and reports `pred_error` (mean cosine distance between predicted and actual next-frame latent).

**Tech Stack:** PyTorch 2.2+, torchvision, Pillow, numpy, einops, requests; managed by uv.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_prepare.py`
- Create: `tests/test_train.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Initialize git**

```bash
git init
```

Expected: `Initialized empty Git repository in ...`

- [ ] **Step 2: Create pyproject.toml**

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
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
]
```

- [ ] **Step 3: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
data/
checkpoint.pt
*.egg-info/
dist/
```

- [ ] **Step 4: Install dependencies**

```bash
uv sync
```

Expected: `.venv/` created, all packages installed without errors.

- [ ] **Step 5: Create test scaffold**

```bash
mkdir -p tests
touch tests/__init__.py tests/test_prepare.py tests/test_train.py tests/test_eval.py
```

- [ ] **Step 6: Verify pytest runs**

```bash
uv run pytest tests/ -v
```

Expected: "no tests ran" or "collected 0 items", exit 0.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore tests/
git commit -m "chore: project scaffold"
```

---

### Task 2: prepare.py — download sprite sheets

**Files:**
- Create: `prepare.py`
- Modify: `tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_prepare.py` with:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_download_sprite_saves_bytes(tmp_path):
    from prepare import download_sprite

    dest = tmp_path / "sprite.png"
    fake_bytes = b"\x89PNG_FAKE"

    mock_resp = MagicMock()
    mock_resp.content = fake_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("prepare.requests.get", return_value=mock_resp) as mock_get:
        download_sprite("https://example.com/sprite.png", dest)
        mock_get.assert_called_once_with("https://example.com/sprite.png", timeout=30)

    assert dest.read_bytes() == fake_bytes


def test_download_sprite_creates_parent_dirs(tmp_path):
    from prepare import download_sprite

    dest = tmp_path / "a" / "b" / "sprite.png"
    mock_resp = MagicMock()
    mock_resp.content = b"data"
    mock_resp.raise_for_status = MagicMock()

    with patch("prepare.requests.get", return_value=mock_resp):
        download_sprite("https://x.com/s.png", dest)

    assert dest.exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_prepare.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'prepare'`

- [ ] **Step 3: Create prepare.py with download_sprite**

Create `prepare.py`:

```python
from pathlib import Path
import requests
from PIL import Image

DATA_DIR = Path("data")
SPRITES_DIR = DATA_DIR / "sprites"
FRAMES_DIR = DATA_DIR / "frames"
FRAME_SIZE = 32  # all frames resized to this square size

# CC0 sprite sheets from OpenGameArt
# (url, name, cols, rows, frame_w, frame_h)
SPRITE_SHEETS = [
    (
        "https://opengameart.org/sites/default/files/Green-Cap-Character-16x18.png",
        "green_cap", 3, 4, 16, 18,
    ),
    (
        "https://opengameart.org/sites/default/files/goblins_sprites.png",
        "goblin", 4, 4, 48, 48,
    ),
]


def download_sprite(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_prepare.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare.py
git commit -m "feat: prepare.py download_sprite"
```

---

### Task 3: prepare.py — slice sprite sheets into frames

**Files:**
- Modify: `prepare.py`
- Modify: `tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_prepare.py`:

```python
from PIL import Image as _Image


def _make_sheet(tmp_path, cols, rows, fw, fh):
    """Synthetic sprite sheet with distinct color per cell."""
    img = _Image.new("RGB", (cols * fw, rows * fh))
    for r in range(rows):
        for c in range(cols):
            color = ((r * cols + c) * 20 % 256, (r * 7) % 256, (c * 13) % 256)
            for y in range(fh):
                for x in range(fw):
                    img.putpixel((c * fw + x, r * fh + y), color)
    path = tmp_path / "sheet.png"
    img.save(path)
    return path


def test_slice_sprite_frame_count(tmp_path):
    from prepare import slice_sprite

    sheet = _make_sheet(tmp_path, cols=3, rows=2, fw=16, fh=16)
    out = tmp_path / "frames" / "hero"
    slice_sprite(sheet, "hero", cols=3, rows=2, frame_w=16, frame_h=16, out_dir=out)
    assert len(list(out.glob("*.png"))) == 6


def test_slice_sprite_naming(tmp_path):
    from prepare import slice_sprite

    sheet = _make_sheet(tmp_path, cols=3, rows=2, fw=16, fh=16)
    out = tmp_path / "frames" / "hero"
    slice_sprite(sheet, "hero", cols=3, rows=2, frame_w=16, frame_h=16, out_dir=out)
    names = {p.name for p in out.glob("*.png")}
    assert "hero_r00_c00.png" in names
    assert "hero_r01_c02.png" in names


def test_slice_sprite_output_size(tmp_path):
    from prepare import slice_sprite, FRAME_SIZE

    sheet = _make_sheet(tmp_path, cols=2, rows=2, fw=24, fh=24)
    out = tmp_path / "frames" / "slime"
    slice_sprite(sheet, "slime", cols=2, rows=2, frame_w=24, frame_h=24, out_dir=out)
    img = _Image.open(out / "slime_r00_c00.png")
    assert img.size == (FRAME_SIZE, FRAME_SIZE)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_prepare.py::test_slice_sprite_frame_count -v
```

Expected: FAIL — `ImportError: cannot import name 'slice_sprite'`

- [ ] **Step 3: Implement slice_sprite in prepare.py**

Append to `prepare.py`:

```python
def slice_sprite(
    sheet_path: Path,
    name: str,
    cols: int,
    rows: int,
    frame_w: int,
    frame_h: int,
    out_dir: Path | None = None,
) -> None:
    if out_dir is None:
        out_dir = FRAMES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(sheet_path).convert("RGB")
    for r in range(rows):
        for c in range(cols):
            box = (c * frame_w, r * frame_h, (c + 1) * frame_w, (r + 1) * frame_h)
            frame = sheet.crop(box).resize((FRAME_SIZE, FRAME_SIZE), Image.NEAREST)
            frame.save(out_dir / f"{name}_r{r:02d}_c{c:02d}.png")
```

- [ ] **Step 4: Add main block to prepare.py**

Append to `prepare.py`:

```python
if __name__ == "__main__":
    print("Downloading sprite sheets...")
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for url, name, cols, rows, frame_w, frame_h in SPRITE_SHEETS:
        dest = SPRITES_DIR / f"{name}.png"
        if not dest.exists():
            print(f"  {name}: downloading from {url}")
            try:
                download_sprite(url, dest)
            except Exception as e:
                print(f"  {name}: FAILED — {e}")
                continue
        else:
            print(f"  {name}: already downloaded, skipping.")
        print(f"  {name}: slicing...")
        slice_sprite(dest, name, cols, rows, frame_w, frame_h)
        n = len(list((FRAMES_DIR / name).glob("*.png")))
        print(f"  {name}: {n} frames saved to {FRAMES_DIR / name}/")

    print("Done.")
```

- [ ] **Step 5: Run all prepare tests**

```bash
uv run pytest tests/test_prepare.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_prepare.py
git commit -m "feat: prepare.py slice_sprite and main block"
```

---

### Task 4: train.py — ViT encoder

**Files:**
- Create: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_train.py` with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_train.py::test_sprite_encoder_output_shape -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Create train.py with SpriteEncoder**

Create `train.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_train.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: train.py SpriteEncoder (ViT)"
```

---

### Task 5: train.py — predictor and Gaussian regularizer

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_train.py::test_sprite_predictor_output_shape -v
```

Expected: FAIL — `ImportError: cannot import name 'SpritePredictor'`

- [ ] **Step 3: Append SpritePredictor and gaussian_reg to train.py**

Append to `train.py` (after SpriteEncoder):

```python
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
```

- [ ] **Step 4: Run all train tests**

```bash
uv run pytest tests/test_train.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: train.py SpritePredictor and gaussian_reg"
```

---

### Task 6: train.py — SpriteFrameDataset

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
def _make_frame_dir(tmp_path, name, rows, cols):
    d = tmp_path / name
    d.mkdir()
    for r in range(rows):
        for c in range(cols):
            Image.new("RGB", (32, 32), color=((r * 50) % 256, (c * 40) % 256, 100)).save(
                d / f"{name}_r{r:02d}_c{c:02d}.png"
            )
    return tmp_path


def test_dataset_pair_count(tmp_path):
    from train import SpriteFrameDataset

    _make_frame_dir(tmp_path, "hero", rows=2, cols=4)
    ds = SpriteFrameDataset(tmp_path)
    assert len(ds) == 6  # 2 rows × 3 consecutive pairs per row


def test_dataset_item_shapes(tmp_path):
    from train import SpriteFrameDataset, FRAME_SIZE

    _make_frame_dir(tmp_path, "hero", rows=1, cols=3)
    ds = SpriteFrameDataset(tmp_path)
    frame_t, frame_t1 = ds[0]
    assert frame_t.shape == (3, FRAME_SIZE, FRAME_SIZE)
    assert frame_t1.shape == (3, FRAME_SIZE, FRAME_SIZE)


def test_dataset_empty_dir(tmp_path):
    from train import SpriteFrameDataset

    ds = SpriteFrameDataset(tmp_path)
    assert len(ds) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_train.py::test_dataset_pair_count -v
```

Expected: FAIL — `ImportError: cannot import name 'SpriteFrameDataset'`

- [ ] **Step 3: Append SpriteFrameDataset to train.py**

Append to `train.py` (after gaussian_reg):

```python
# ── Dataset ────────────────────────────────────────────────────────────────
class SpriteFrameDataset(torch.utils.data.Dataset):
    _transform = transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def __init__(self, frames_root: Path):
        self.pairs: list[tuple[Path, Path]] = []
        groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for p in frames_root.rglob("*_r[0-9][0-9]_c[0-9][0-9].png"):
            base, col_str = p.stem.rsplit("_c", 1)
            groups[base].append((int(col_str), p))
        for frames in groups.values():
            frames.sort()
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i][1], frames[i + 1][1]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p_t, p_t1 = self.pairs[idx]
        return (
            self._transform(Image.open(p_t).convert("RGB")),
            self._transform(Image.open(p_t1).convert("RGB")),
        )
```

- [ ] **Step 4: Run all train tests**

```bash
uv run pytest tests/test_train.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: train.py SpriteFrameDataset"
```

---

### Task 7: train.py — JEPA training loop

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
def test_jepa_training_step_produces_finite_loss():
    from train import SpriteEncoder, SpritePredictor, gaussian_reg, LAMBDA_REG

    encoder = SpriteEncoder()
    predictor = SpritePredictor()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4
    )
    frame_t = torch.randn(8, 3, 32, 32)
    frame_t1 = torch.randn(8, 3, 32, 32)

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
```

- [ ] **Step 2: Run to verify it passes (no new code needed — tests existing components)**

```bash
uv run pytest tests/test_train.py::test_jepa_training_step_produces_finite_loss -v
```

Expected: PASS. This confirms all components integrate correctly.

- [ ] **Step 3: Append training loop to train.py**

Append to `train.py` (after SpriteFrameDataset):

```python
# ── Training loop ──────────────────────────────────────────────────────────
def train(steps: int = MAX_STEPS, batch_size: int = BATCH_SIZE, device: str = "cuda") -> None:
    dev = torch.device(device)
    dataset = SpriteFrameDataset(FRAMES_DIR)
    if len(dataset) == 0:
        raise RuntimeError(f"No frame pairs found in {FRAMES_DIR}. Run prepare.py first.")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    step, best_loss = 0, float("inf")
    data_iter = iter(loader)
    print(f"Training on {len(dataset)} frame pairs  device={device}")

    while step < steps:
        try:
            frame_t, frame_t1 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frame_t, frame_t1 = next(data_iter)

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
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, device=args.device)
```

- [ ] **Step 4: Run all train tests**

```bash
uv run pytest tests/test_train.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: train.py JEPA training loop"
```

---

### Task 8: eval.py — pred_error metric

**Files:**
- Create: `eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_eval.py` with:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_eval.py::test_compute_pred_error_returns_float -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: Create eval.py**

Create `eval.py`:

```python
from pathlib import Path
import torch
import torch.nn.functional as F
from train import SpriteEncoder, SpritePredictor, SpriteFrameDataset


def compute_pred_error(
    checkpoint: Path,
    frames_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    dev = torch.device(device)
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()

    loader = torch.utils.data.DataLoader(
        SpriteFrameDataset(frames_dir), batch_size=batch_size, shuffle=False
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for frame_t, frame_t1 in loader:
            frame_t, frame_t1 = frame_t.to(dev), frame_t1.to(dev)
            z_t1_pred = predictor(encoder(frame_t))
            z_t1 = encoder(frame_t1)
            total += (1 - F.cosine_similarity(z_t1_pred, z_t1)).sum().item()
            n += frame_t.shape[0]
    return total / n if n > 0 else 0.0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    p.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)

    error = compute_pred_error(args.checkpoint, args.frames_dir, device=args.device)
    print(f"pred_error (mean cosine distance): {error:.4f}")
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eval.py tests/test_eval.py
git commit -m "feat: eval.py compute_pred_error"
```

---

## End state

After all tasks complete, the repo looks like:

```
prepare.py     — download + slice sprites from OpenGameArt
train.py       — ViT encoder, MLP predictor, JEPA loop
eval.py        — pred_error metric
pyproject.toml
tests/
  test_prepare.py
  test_train.py
  test_eval.py
docs/
  IDEA.md
  superpowers/plans/2026-05-17-piccolo-spritesheet.md
```

Quick start works:

```bash
uv sync
uv run prepare.py     # download sprites, slice frames
uv run train.py       # train JEPA on GPU
uv run eval.py        # pred_error: 0.xxxx
```
