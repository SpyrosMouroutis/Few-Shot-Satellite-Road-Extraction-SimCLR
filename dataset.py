import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def _list_files(folder: str, max_samples: int | None = None) -> list[str]:
    paths = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                paths.append(entry.path)
                if max_samples is not None and len(paths) >= max_samples:
                    break
    return sorted(paths)

class SimCLRDataset(Dataset):
    def __init__(self, img_dir: str, img_size: int = 256, max_samples: int | None = 10000):
        self.paths = _list_files(img_dir, max_samples=max_samples)
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)

class SegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_size: int = 256,
        fraction: float = 1.0,
        augment: bool = True,
        seed: int = 42,
    ):
        all_imgs = _list_files(img_dir)
        all_masks = _list_files(mask_dir)
        assert len(all_imgs) == len(all_masks), "Image and mask counts do not match."

        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            n_keep = max(1, int(len(all_imgs) * fraction))
            indices = sorted(rng.choice(len(all_imgs), size=n_keep, replace=False))
            all_imgs = [all_imgs[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]

        self.imgs = all_imgs
        self.masks = all_masks
        self.img_size = img_size
        self.augment = augment
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.imgs[idx]).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(self.masks[idx]).convert("L").resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )

        if self.augment:
            if torch.rand(1) > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if torch.rand(1) > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)

        img_t = self.norm(TF.to_tensor(img))
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        return img_t, mask_t