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
