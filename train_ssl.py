import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

from dataset import SimCLRDataset

class SimCLRModel(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return F.normalize(self.projector(h), dim=1)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(sim, targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/unlabeled")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--out", type=str, default="encoder_ssl.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SimCLRDataset(args.data_dir, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(ds)} unlabeled tiles ({len(loader)} batches/epoch) on {device}.", flush=True)

    model = SimCLRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (x1, x2) in enumerate(loader, 1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(
                    f"  [Epoch {epoch:02d}/{args.epochs:02d}] Batch {step}/{len(loader)} "
                    f"| Loss: {running_loss / step:.4f}",
                    flush=True,
                )

        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] Avg Loss: {running_loss / len(loader):.4f} "
            f"| Time: {elapsed:.1f}s",
            flush=True,
        )

    torch.save(model.encoder.state_dict(), args.out)
    print(f"Saved pre-trained ResNet-18 encoder weights to {args.out}", flush=True)

if __name__ == "__main__":
    main()