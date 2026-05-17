from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from PIL import Image
from train import (
    SpriteEncoder,
    SpritePredictor,
    _frame_transform,
    CHECKPOINT_PATH,
    FRAMES_DIR,
    _get_device,
)


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[SpriteEncoder, SpritePredictor]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder = SpriteEncoder().to(device)
    predictor = SpritePredictor().to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor


def build_latent_index(
    encoder: SpriteEncoder, frames_dir: Path, device: torch.device
) -> tuple[torch.Tensor, list[Path]]:
    paths = sorted(frames_dir.rglob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    latents = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGBA")
            x = _frame_transform(img).unsqueeze(0).to(device)
            latents.append(encoder(x).squeeze(0))
    return torch.stack(latents), paths


def find_nearest(
    z_pred: torch.Tensor, latent_stack: torch.Tensor, frame_paths: list[Path]
) -> Path:
    sims = F.cosine_similarity(z_pred.unsqueeze(0), latent_stack, dim=1)
    return frame_paths[sims.argmax().item()]


def encode_frame(
    path: Path, encoder: SpriteEncoder, device: torch.device
) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    x = _frame_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(x).squeeze(0)


def predict_next(
    z_t: torch.Tensor,
    predictor: SpritePredictor,
    latent_stack: torch.Tensor,
    frame_paths: list[Path],
) -> tuple[torch.Tensor, Path]:
    with torch.no_grad():
        z_t1_pred = predictor(z_t.unsqueeze(0)).squeeze(0)
    nearest = find_nearest(z_t1_pred, latent_stack, frame_paths)
    return z_t1_pred, nearest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sprite frame prediction via NN retrieval")
    p.add_argument("--input", type=Path, required=True, help="Input frame (PNG)")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR,
                   help="Dataset frames directory for building latent index")
    p.add_argument("--output", type=Path, default=Path("predicted.png"),
                   help="Output path (single-frame mode)")
    p.add_argument("--frame-count", type=int, default=None,
                   help="Predict this many frames (sequence mode)")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="Output directory (sequence mode)")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    dev = torch.device(_get_device(args.device))
    encoder, predictor = load_model(args.checkpoint, dev)
    print(f"Building latent index from {args.frames_dir} ...")
    latent_stack, frame_paths = build_latent_index(encoder, args.frames_dir, dev)
    print(f"  Indexed {len(frame_paths)} frames.")

    if args.frame_count is None:
        z_t = encode_frame(args.input, encoder, dev)
        _, nearest = predict_next(z_t, predictor, latent_stack, frame_paths)
        shutil.copy2(nearest, args.output)
        print(f"Predicted next frame → {args.output}  (source: {nearest})")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input, args.output_dir / "frame_000.png")
        print(f"frame_000.png  (input)")
        current_path = args.input
        for i in range(1, args.frame_count + 1):
            z_t = encode_frame(current_path, encoder, dev)
            _, nearest = predict_next(z_t, predictor, latent_stack, frame_paths)
            out_name = f"frame_{i:03d}.png"
            shutil.copy2(nearest, args.output_dir / out_name)
            print(f"{out_name}  (source: {nearest})")
            current_path = nearest
        print(f"Sequence saved to {args.output_dir}/")
