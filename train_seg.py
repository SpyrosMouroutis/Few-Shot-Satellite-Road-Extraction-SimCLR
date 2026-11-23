import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model_unet import ResNet18UNet

def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()
    return 0.5 * bce + 0.5 * dice_loss

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    inter_total = 0.0
    union_total = 0.0
    pred_total = 0.0
    targ_total = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()

        inter_total += (preds * masks).sum().item()
        union_total += ((preds + masks) > 0).float().sum().item()
        pred_total += preds.sum().item()
        targ_total += masks.sum().item()

    iou = (inter_total + 1e-6) / (union_total + 1e-6)
    dice = (2.0 * inter_total + 1e-6) / (pred_total + targ_total + 1e-6)
    return iou, dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img", type=str, default="data/train/images")
    parser.add_argument("--train_mask", type=str, default="data/train/masks")
    parser.add_argument("--val_img", type=str, default="data/val/images")
    parser.add_argument("--val_mask", type=str, default="data/val/masks")
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--ssl_weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="unet_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SegmentationDataset(args.train_img, args.train_mask, fraction=args.fraction, augment=True)
    val_ds = SegmentationDataset(args.val_img, args.val_mask, fraction=1.0, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    tag = "SimCLR" if args.ssl_weights else "Random-Init"
    print(f"[{tag} | fraction={args.fraction:.0%}] Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}", flush=True)

    model = ResNet18UNet(ssl_weights_path=args.ssl_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        val_iou, val_dice = evaluate(model, val_loader, device)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.out)

        print(
            f"  [Epoch {epoch:02d}/{args.epochs:02d}] Loss: {running_loss / len(train_loader):.4f} "
            f"| Val IoU: {val_iou * 100:.2f}% | Val Dice: {val_dice * 100:.2f}% "
            f"| Best IoU: {best_iou * 100:.2f}% | {time.time() - t0:.1f}s",
            flush=True,
        )

if __name__ == "__main__":
    main()