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
