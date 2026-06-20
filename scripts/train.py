#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mn5_segmentation.data import KvasirSegDataset, find_dataset_source
from mn5_segmentation.metrics import dice_score_from_logits, iou_score_from_logits
from mn5_segmentation.model import UNet
from mn5_segmentation.utils import read_json, seed_everything, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-file", default="outputs/splits/kvasir_split.json")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unet_kvasir.pt")
    parser.add_argument("--history", default="outputs/checkpoints/history.json")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--gpus", type=int, default=1, help="Use 1 or 2 GPUs when available.")
    return parser.parse_args()


def build_model(requested_gpus: int):
    if requested_gpus < 1:
        raise ValueError("--gpus must be at least 1")

    available_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpus = min(requested_gpus, available_gpus)
    model = UNet().to(device)

    if use_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(use_gpus)))
        print(f"Training with DataParallel on {use_gpus} GPUs")
    elif use_gpus == 1:
        print("Training on one GPU")
    else:
        print("CUDA is not available. Training on CPU.")

    return model, device


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks, _keys in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score_from_logits(logits.detach(), masks).item() * batch_size
        total_iou += iou_score_from_logits(logits.detach(), masks).item() * batch_size

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    args = parse_args()
    seed_everything(args.seed)

    split = read_json(args.split_file)
    source = find_dataset_source(args.data_root)
    train_dataset = KvasirSegDataset(source, split["splits"]["train"], args.image_size)
    val_dataset = KvasirSegDataset(source, split["splits"]["val"], args.image_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model, device = build_model(args.gpus)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} iou {train_iou:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} iou {val_iou:.4f}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        },
        checkpoint_path,
    )
    write_json(history, args.history)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved history to {args.history}")


if __name__ == "__main__":
    main()

