from pathlib import Path
import torch
import torch.nn.functional as F
from train import SpriteEncoder, SpritePredictor, SpriteDecoder, SpriteFrameDataset


def compute_pred_error(
    checkpoint: Path,
    frames_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    dev = torch.device(device)
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    decoder = SpriteDecoder().to(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    predictor.eval()
    decoder.eval()

    loader = torch.utils.data.DataLoader(
        SpriteFrameDataset(frames_dir), batch_size=batch_size, shuffle=False
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for frame_t, frame_t1, _texts in loader:
            frame_t, frame_t1 = frame_t.to(dev), frame_t1.to(dev)
            frame_t1_gen = decoder(predictor(encoder(frame_t)))
            total += F.l1_loss(frame_t1_gen, frame_t1, reduction="sum").item()
            n += frame_t.shape[0] * frame_t.shape[1] * frame_t.shape[2] * frame_t.shape[3]
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
    print(f"pred_error (mean pixel L1): {error:.4f}")
