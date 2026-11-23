import random
from pathlib import Path
from PIL import Image
import numpy as np

def tile_pair(
    img_path: Path,
    mask_path: Path | None,
    out_img_dir: Path,
    out_mask_dir: Path | None,
    tile_size: int = 256,
):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path else None
    w, h = img.size

    idx = 0
    for top in range(0, h - tile_size + 1, tile_size):
        for left in range(0, w - tile_size + 1, tile_size):
            box = (left, top, left + tile_size, top + tile_size)
            crop_img = img.crop(box)

            if mask is not None and out_mask_dir is not None:
                crop_mask = mask.crop(box)
                arr = (np.array(crop_mask) > 128).astype(np.uint8) * 255
                if arr.mean() < 2.0:
                    continue
                Image.fromarray(arr).save(out_mask_dir / f"{img_path.stem}_{idx}.png")

            crop_img.save(out_img_dir / f"{img_path.stem}_{idx}.jpg")
            idx += 1

def build_dataset():
    dirs = [
        "data/unlabeled",
        "data/train/images",
        "data/train/masks",
        "data/val/images",
        "data/val/masks",
        "data/mass_test/images",
        "data/mass_test/masks",
    ]
    for p in dirs:
        Path(p).mkdir(parents=True, exist_ok=True)

    dg_sats = sorted(Path("raw_deepglobe").rglob("*_sat.jpg"))
    labeled_sats = [
        sat for sat in dg_sats if sat.with_name(sat.name.replace("_sat.jpg", "_mask.png")).exists()
    ]
    unlabeled_sats = [
        sat for sat in dg_sats if not sat.with_name(sat.name.replace("_sat.jpg", "_mask.png")).exists()
    ]

    print(f"Found {len(labeled_sats)} labeled and {len(unlabeled_sats)} unlabeled DeepGlobe images.")

    random.seed(42)
    random.shuffle(labeled_sats)
    split = int(0.85 * len(labeled_sats))

    for sat in labeled_sats[:split]:
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        tile_pair(sat, mask, Path("data/train/images"), Path("data/train/masks"))
        tile_pair(sat, None, Path("data/unlabeled"), None)

    for sat in labeled_sats[split:]:
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        tile_pair(sat, mask, Path("data/val/images"), Path("data/val/masks"))

    for sat in unlabeled_sats:
        tile_pair(sat, None, Path("data/unlabeled"), None)

    mass_test_imgs = sorted(Path("raw_mass").rglob("test/*.tif*"))
    print(f"Found {len(mass_test_imgs)} Massachusetts test images.")

    for img in mass_test_imgs:
        label_dir = img.parent.parent / "test_labels"
        candidates = list(label_dir.glob(f"{img.stem}.tif*"))
        if candidates:
            tile_pair(img, candidates[0], Path("data/mass_test/images"), Path("data/mass_test/masks"))

    print("Tiling complete:")
    for d in ["data/unlabeled", "data/train/images", "data/val/images", "data/mass_test/images"]:
        print(f"  {d}: {len(list(Path(d).glob('*.*')))} tiles")


if __name__ == "__main__":
    build_dataset()