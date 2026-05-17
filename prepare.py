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
