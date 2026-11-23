import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = min(num_groups, channels)
            while channels % groups != 0:
                groups //= 2
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            replace_bn_with_gn(child, num_groups)
    return module

class SimCLRModel(nn.Module):
    def __init__(self, proj_dim: int = 128, use_groupnorm: bool = False):
        super().__init__()
        self.encoder = resnet18(weights=None)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        if use_groupnorm:
            replace_bn_with_gn(self.encoder)

        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return F.normalize(self.projector(h), dim=-1)

class UpBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class ResNetUNet(nn.Module):
    def __init__(
        self,
        init_mode: str = "imagenet",
        ssl_ckpt_path: str | None = None,
        use_groupnorm: bool = False,
    ):
        super().__init__()
        if init_mode == "imagenet":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        if use_groupnorm:
            replace_bn_with_gn(backbone)

        if init_mode == "ssl":
            assert ssl_ckpt_path is not None, "Provide ssl_ckpt_path for init_mode='ssl'"
            state_dict = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict, strict=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) 
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1 
        self.enc2 = backbone.layer2  
        self.enc3 = backbone.layer3  
        self.enc4 = backbone.layer4 

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

        if use_groupnorm:
            replace_bn_with_gn(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        out = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(out)