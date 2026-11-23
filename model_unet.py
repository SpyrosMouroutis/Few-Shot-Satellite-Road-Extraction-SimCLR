import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet18UNet(nn.Module):
    def __init__(self, ssl_weights_path: str | None = None):
        super().__init__()
        backbone = models.resnet18(weights=None)
        if ssl_weights_path is not None:
            state = torch.load(ssl_weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            print(
                f"Loaded SSL encoder from {ssl_weights_path} (missing={missing}, unexpected={unexpected})",
                flush=True,
            )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64 ch
        self.pool = backbone.maxpool                                           # /4
        self.enc1 = backbone.layer1                                            # /4, 64 ch
        self.enc2 = backbone.layer2                                            # /8, 128 ch
        self.enc3 = backbone.layer3                                            # /16, 256 ch
        self.enc4 = backbone.layer4                                            # /32, 512 ch

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        return self.final_up(d1)