#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from mn5_segmentation.data import load_image_tensor
from mn5_segmentation.model import UNet, strip_dataparallel_prefix


def load_model(checkpoint_path: str | Path, device: str | torch.device | None = None) -> tuple[UNet, torch.device]:
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = UNet().to(device)
    model.load_state_dict(strip_dataparallel_prefix(state_dict))
    model.eval()
    return model, device


def predict_mask(model, image_path: str | Path, device, image_size: int = 256, threshold: float = 0.5):
    tensor = load_image_tensor(image_path, image_size=image_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probability > threshold).astype(np.uint8) * 255
    return probability, mask


def save_prediction(mask, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def show_prediction(image_path: str | Path, probability, mask):
    image = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(probability, cmap="viridis")
    axes[1].set_title("probability")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("mask")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on one image.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unet_kvasir.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/prediction.png")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model, device = load_model(args.checkpoint, args.device)
    probability, mask = predict_mask(model, args.image, device, args.image_size, args.threshold)
    save_prediction(mask, args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()

