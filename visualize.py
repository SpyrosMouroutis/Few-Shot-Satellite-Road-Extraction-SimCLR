import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import SegmentationDataset
from model_unet import ResNet18UNet

def denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_t.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img", type=str, default="data/val/images")
    parser.add_argument("--val_mask", type=str, default="data/val/masks")
    parser.add_argument("--rand_weights", type=str, default="unet_rand_1pct.pt")
    parser.add_argument("--ssl_weights", type=str, default="unet_ssl_1pct.pt")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--out", type=str, default="comparison_1pct.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = SegmentationDataset(args.val_img, args.val_mask, fraction=1.0, augment=False)

    model_rand = ResNet18UNet().to(device)
    model_rand.load_state_dict(torch.load(args.rand_weights, map_location=device, weights_only=True))
    model_rand.eval()

    model_ssl = ResNet18UNet().to(device)
    model_ssl.load_state_dict(torch.load(args.ssl_weights, map_location=device, weights_only=True))
    model_ssl.eval()

    chosen = []
    for idx in range(0, len(val_ds), 17):
        _, mask = val_ds[idx]
        if mask.mean().item() > 0.04:
            chosen.append(idx)
            if len(chosen) == args.num_samples:
                break

    fig, axes = plt.subplots(args.num_samples, 4, figsize=(14, 3.5 * args.num_samples))
    headers = ["Satellite Tile", "Ground Truth Mask", "Random-Init U-Net", "SimCLR U-Net (Ours)"]
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row, idx in enumerate(chosen):
        img_t, mask_t = val_ds[idx]
        inp = img_t.unsqueeze(0).to(device)

        pred_rand = (torch.sigmoid(model_rand(inp))[0, 0] > 0.5).cpu().numpy()
        pred_ssl = (torch.sigmoid(model_ssl(inp))[0, 0] > 0.5).cpu().numpy()

        axes[row, 0].imshow(denormalize(img_t))
        axes[row, 1].imshow(mask_t[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred_rand, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(pred_ssl, cmap="gray", vmin=0, vmax=1)

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved side-by-side comparison grid to {args.out}")

if __name__ == "__main__":
    main()