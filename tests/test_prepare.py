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
    from prepare import slice_sprite

    sheet = _make_sheet(tmp_path, cols=2, rows=2, fw=24, fh=24)
    out = tmp_path / "frames" / "slime"
    slice_sprite(sheet, "slime", cols=2, rows=2, frame_w=24, frame_h=24, out_dir=out)
    img = _Image.open(out / "slime_r00_c00.png")
    assert img.size == (24, 24)
    assert img.mode == "RGBA"


# ── HuggingFace spraix helpers ─────────────────────────────────────────────

def test_parse_frame_count_standard():
    from prepare import _parse_frame_count

    assert _parse_frame_count("8-frame sprite animation of: hero") == 8
    assert _parse_frame_count("17-frame sprite animation of: hobbit") == 17
    assert _parse_frame_count("32 frame walk cycle") == 32


def test_parse_frame_count_missing():
    from prepare import _parse_frame_count

    assert _parse_frame_count("sprite animation of: hero") is None
    assert _parse_frame_count("") is None


def _make_hf_sheet(n_frames: int, frame_w: int, frame_h: int, bg=(180, 180, 180, 255)):
    """Synthetic horizontal-strip sprite sheet with gray border padding to 1024×1024."""
    content_w = n_frames * frame_w
    full = _Image.new("RGBA", (1024, 1024), color=bg)
    for i in range(n_frames):
        color = ((i * 40) % 256, 100, 200, 255)
        cell = _Image.new("RGBA", (frame_w, frame_h), color=color)
        full.paste(cell, (i * frame_w, 0))
    return full


def test_content_bbox_detects_sprite_region():
    from prepare import _content_bbox

    img = _make_hf_sheet(n_frames=8, frame_w=64, frame_h=128)
    x0, y0, x1, y1 = _content_bbox(img)
    assert x0 == 0
    assert y0 == 0
    assert x1 == 8 * 64   # 512
    assert y1 == 128


def test_content_bbox_full_image_when_no_content():
    from prepare import _content_bbox

    img = _Image.new("RGBA", (64, 64), color=(180, 180, 180, 255))
    x0, y0, x1, y1 = _content_bbox(img)
    assert (x0, y0, x1, y1) == (0, 0, 64, 64)


def test_slice_hf_item_frame_count(tmp_path):
    from prepare import slice_hf_item

    img = _make_hf_sheet(n_frames=6, frame_w=48, frame_h=64)
    text = "6-frame sprite animation of: slime"
    n = slice_hf_item(img, text, "test_sprite", tmp_path / "test_sprite")
    assert n == 6
    frames = list((tmp_path / "test_sprite").glob("*.png"))
    assert len(frames) == 6


def test_slice_hf_item_naming(tmp_path):
    from prepare import slice_hf_item

    img = _make_hf_sheet(n_frames=4, frame_w=32, frame_h=32)
    slice_hf_item(img, "4-frame sprite animation of: wizard", "wiz", tmp_path / "wiz")
    names = {p.name for p in (tmp_path / "wiz").glob("*.png")}
    assert "wiz_r00_c00.png" in names
    assert "wiz_r00_c03.png" in names


def test_slice_hf_item_writes_metadata(tmp_path):
    import json as _json
    from prepare import slice_hf_item

    text = "5-frame sprite animation of: knight"
    img = _make_hf_sheet(n_frames=5, frame_w=40, frame_h=40)
    slice_hf_item(img, text, "knight", tmp_path / "knight")
    meta = _json.loads((tmp_path / "knight" / "metadata.json").read_text())
    assert meta["text"] == text
    assert meta["n_frames"] == 5


def test_slice_hf_item_skips_single_frame(tmp_path):
    from prepare import slice_hf_item

    img = _make_hf_sheet(n_frames=1, frame_w=64, frame_h=64)
    n = slice_hf_item(img, "1-frame sprite animation", "single", tmp_path / "single")
    assert n == 0


def test_slice_hf_item_skips_no_frame_count(tmp_path):
    from prepare import slice_hf_item

    img = _Image.new("RGBA", (1024, 1024))
    n = slice_hf_item(img, "sprite animation of: hero", "hero", tmp_path / "hero")
    assert n == 0
