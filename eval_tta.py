import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model_unet import ResNet18UNet

@torch.no_grad()
def predict_tta(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """4x Test-Time Augmentation: original, h-flip, v-flip, 90-deg rotation."""
    p0 = torch.sigmoid(model(imgs))

    p_h = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
    p_h = torch.flip(p_h, dims=[3])

    p_v = torch.sigmoid(model(torch.flip(imgs, dims=[2])))
    p_v = torch.flip(p_v, dims=[2])

    p_r = torch.sigmoid(model(torch.rot90(imgs, k=1, dims=[2, 3])))
    p_r = torch.rot90(p_r, k=3, dims=[2, 3])

    return 0.25 * (p0 + p_h + p_v + p_r)

@torch.no_grad()
def evaluate_tta_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> tuple[float, float, float]:
    model.eval()
    inter = torch.zeros(len(thresholds), dtype=torch.float64)
    union = torch.zeros(len(thresholds), dtype=torch.float64)
    pred_sum = torch.zeros(len(thresholds), dtype=torch.float64)
    target_sum = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            probs = predict_tta(model, imgs).cpu()

        target = masks > 0.5
        target_sum += target.sum().item()

        for i, thr in enumerate(thresholds):
            pred = probs > thr
            inter[i] += (pred & target).sum().item()
            union[i] += (pred | target).sum().item()
            pred_sum[i] += pred.sum().item()

    ious = (inter + 1e-6) / (union + 1e-6)
    dices = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)

    best_idx = int(torch.argmax(ious).item())
    return thresholds[best_idx], ious[best_idx].item() * 100.0, dices[best_idx].item() * 100.0

@torch.no_grad()
def save_comparison(
    model_rand: torch.nn.Module,
    model_ssl: torch.nn.Module,
    val_ds: SegmentationDataset,
    device: torch.device,
    thr_rand: float,
    thr_ssl: float,
    out_path: str,
    num_samples: int = 4,
) -> None:
    model_rand.eval()
    model_ssl.eval()

    samples = []
    for idx in range(len(val_ds)):
        img, mask = val_ds[idx]
        if mask.mean().item() > 0.04:
            samples.append((img, mask))
            if len(samples) == num_samples:
                break

    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples), dpi=150)
    titles = [
        "Satellite Tile",
        "Ground Truth Mask",
        f"Random-Init + TTA (t={thr_rand:.2f})",
        f"SimCLR + TTA (t={thr_ssl:.2f})",
    ]

    for row, (img, mask) in enumerate(samples):
        inp = img.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_rand = (predict_tta(model_rand, inp)[0, 0].cpu() > thr_rand).float()
            pred_ssl = (predict_tta(model_ssl, inp)[0, 0].cpu() > thr_ssl).float()

        panels = [img.permute(1, 2, 0).numpy(), mask[0].numpy(), pred_rand.numpy(), pred_ssl.numpy()]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel if col == 0 else panel, cmap=None if col == 0 else "gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved TTA comparison grid to {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rand_weights", type=str, required=True)
    parser.add_argument("--ssl_weights", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = SegmentationDataset(f"{args.data_dir}/val/images", f"{args.data_dir}/val/masks", augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    model_rand = ResNet18UNet().to(device)
    model_rand.load_state_dict(torch.load(args.rand_weights, map_location=device, weights_only=True))
    thr_r, iou_r, dice_r = evaluate_tta_sweep(model_rand, val_loader, device, thresholds)
    print(f"[Random-Init + 4x TTA] Best Thr: {thr_r:.2f} | Val IoU: {iou_r:.2f}% | Val Dice: {dice_r:.2f}%")

    model_ssl = ResNet18UNet().to(device)
    model_ssl.load_state_dict(torch.load(args.ssl_weights, map_location=device, weights_only=True))
    thr_s, iou_s, dice_s = evaluate_tta_sweep(model_ssl, val_loader, device, thresholds)
    print(f"[SimCLR      + 4x TTA] Best Thr: {thr_s:.2f} | Val IoU: {iou_s:.2f}% | Val Dice: {dice_s:.2f}%")

    save_comparison(model_rand, model_ssl, val_ds, device, thr_r, thr_s, args.out)

if __name__ == "__main__":
    main()