from pathlib import Path
import json
import re
import requests
import numpy as np
from PIL import Image

DATA_DIR = Path("data")
SPRITES_DIR = DATA_DIR / "sprites"
FRAMES_DIR = DATA_DIR / "frames"

# CC0 sprite sheets from OpenGameArt
# (url, name, cols, rows, frame_w, frame_h)
SPRITE_SHEETS = [
    (
        "https://opengameart.org/sites/default/files/Green-Cap-Character-16x18.png",
        "green_cap", 3, 4, 16, 18,
    ),
    (
        "https://opengameart.org/sites/default/files/Hero.png",
        "Hero-01", 8, 3, 16, 16,
    ),
]


def download_sprite(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


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
    sheet = Image.open(sheet_path).convert("RGBA")
    for r in range(rows):
        for c in range(cols):
            box = (c * frame_w, r * frame_h, (c + 1) * frame_w, (r + 1) * frame_h)
            frame = sheet.crop(box)
            frame.save(out_dir / f"{name}_r{r:02d}_c{c:02d}.png")


def _parse_frame_count(text: str) -> int | None:
    """Extract N from 'N-frame sprite animation ...' description."""
    m = re.search(r"(\d+)\s*-?\s*frame", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _content_bbox(img: Image.Image, bg_tolerance: int = 15) -> tuple[int, int, int, int]:
    """
    Return (x0, y0, x1, y1) bounding box of non-background pixels.
    Background colour is inferred from the bottom-right corner pixel (reliably
    background for horizontal-strip sprite sheets padded to a square canvas).
    Falls back to the full image if no content is detected.
    """
    arr = np.array(img.convert("RGBA"), dtype=np.int32)
    bg = arr[-1, -1]
    content = np.abs(arr - bg).max(axis=2) > bg_tolerance
    rows_with = np.where(content.any(axis=1))[0]
    cols_with = np.where(content.any(axis=0))[0]
    if len(rows_with) == 0 or len(cols_with) == 0:
        return 0, 0, img.width, img.height
    return int(cols_with[0]), int(rows_with[0]), int(cols_with[-1]) + 1, int(rows_with[-1]) + 1


def slice_hf_item(
    image: Image.Image,
    text: str,
    name: str,
    out_dir: Path,
) -> int:
    """
    Slice one spraix sprite sheet (horizontal strip, uniform bg) into N frames.
    Reads pixels to detect the content bounding box; uses N from text to divide
    the content width.  Returns the number of frames saved, or 0 if skipped.
    """
    n_frames = _parse_frame_count(text)
    if n_frames is None or n_frames < 2:
        return 0

    img = image.convert("RGBA")
    x0, y0, x1, y1 = _content_bbox(img)
    frame_w = (x1 - x0) // n_frames
    if frame_w < 1:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        box = (x0 + i * frame_w, y0, x0 + (i + 1) * frame_w, y1)
        img.crop(box).save(out_dir / f"{name}_r00_c{i:02d}.png")

    (out_dir / "metadata.json").write_text(
        json.dumps({"text": text, "n_frames": n_frames}), encoding="utf-8"
    )
    return n_frames


def download_spraix_1024(frames_dir: Path = FRAMES_DIR) -> None:
    """Download and slice pawkanarek/spraix_1024 from HuggingFace."""
    from datasets import load_dataset

    print("Loading spraix_1024 from HuggingFace (560 entries)...")
    ds = load_dataset("pawkanarek/spraix_1024", split="train")
    total_frames = 0
    for idx, item in enumerate(ds):
        name = f"spraix_{idx:05d}"
        out_dir = frames_dir / name
        if out_dir.exists():
            continue
        n = slice_hf_item(item["image"], item["text"], name, out_dir)
        if n:
            total_frames += n
            if idx % 50 == 0:
                print(f"  {idx}/{len(ds)}: {name} → {n} frames")
        else:
            print(f"  {idx}/{len(ds)}: {name}: skipped (no frame count in text)")
    print(f"spraix_1024: {total_frames} frames saved to {frames_dir}/")


if __name__ == "__main__":
    print("Downloading OGA sprite sheets...")
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

    download_spraix_1024()
    print("Done.")
