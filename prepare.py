from pathlib import Path
from collections import defaultdict
import requests
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


def download_loacky_sprites(frames_dir: Path = FRAMES_DIR) -> None:
    """Download Loacky/sprite-animation from HuggingFace.

    Frames are already individually sliced and aligned; we just group them by
    label (animation sequence) and write them in our naming convention:
        {label}_r00_c{frame_index:02d}.png
    """
    from datasets import load_dataset

    print("Loading Loacky/sprite-animation from HuggingFace (1290 frames)...")
    ds = load_dataset("Loacky/sprite-animation", split="train")
    label_names: list[str] = ds.features["label"].names

    groups: dict[str, list] = defaultdict(list)
    for item in ds:
        groups[label_names[item["label"]]].append(item["image"])

    total = 0
    for label, frames in groups.items():
        out_dir = frames_dir / label
        if out_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(frames):
            img.convert("RGBA").save(out_dir / f"{label}_r00_c{idx:02d}.png")
        total += len(frames)
        print(f"  {label}: {len(frames)} frames")

    print(f"Loacky/sprite-animation: {total} frames saved to {frames_dir}/")


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

    download_loacky_sprites()
    print("Done.")
