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
