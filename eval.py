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
        for frame_t, frame_t1, _texts in loader:
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
