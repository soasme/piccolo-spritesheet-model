from pathlib import Path
import torch
import numpy as np
from PIL import Image
from train import SpriteEncoder, SpritePredictor, SpriteDecoder, _frame_transform, _get_device, CHECKPOINT_PATH


def predict_next_frame(
    input_path: Path,
    output_path: Path,
    checkpoint: Path = CHECKPOINT_PATH,
    device: str = "auto",
    steps: int = 1,
) -> None:
    dev = torch.device(_get_device(device))
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    encoder = SpriteEncoder().to(dev)
    predictor = SpritePredictor().to(dev)
    decoder = SpriteDecoder().to(dev)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    predictor.eval()
    decoder.eval()

    frame = _frame_transform(Image.open(input_path).convert("RGBA")).unsqueeze(0).to(dev)
    with torch.no_grad():
        z = encoder(frame)
        for _ in range(steps):
            z = predictor(z)
        out = decoder(z)

    # Denormalize from [-1, 1] to [0, 255] uint8
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
    arr = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGBA").save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="input frame (PNG/RGBA)")
    p.add_argument("output", type=Path, help="output frame path")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--steps", type=int, default=1, help="number of prediction steps into the future")
    args = p.parse_args()
    predict_next_frame(args.input, args.output, args.checkpoint, args.device, args.steps)
