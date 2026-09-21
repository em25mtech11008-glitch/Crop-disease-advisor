"""
Data preprocessing pipeline for PlantVillage dataset.
Prompt 1.3 — Data Preprocessing Pipeline
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

import torch
from torchvision import transforms, datasets


# ── Transforms ────────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Dataset Split ─────────────────────────────────────────────────────────────

def collect_image_paths(raw_dir: str):
    """Walk raw_dir (ImageFolder layout) → list of (path, class_name)."""
    raw_path = Path(raw_dir)
    samples  = []
    classes  = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])

    for cls in classes:
        cls_dir = raw_path / cls
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((str(img_path), cls))

    return samples, classes


def stratified_split(samples, train=0.8, val=0.1, test=0.1, seed=42):
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1."
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths, labels, test_size=test, stratify=labels, random_state=seed
    )
    val_ratio = val / (train + val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=seed
    )

    return (
        list(zip(X_train, y_train)),
        list(zip(X_val,   y_val)),
        list(zip(X_test,  y_test)),
    )


def copy_split_to_output(split_samples, split_name: str, output_dir: str):
    """Copy files to output_dir/{split_name}/{class_name}/."""
    for src_path, cls_name in split_samples:
        dest_dir = Path(output_dir) / split_name / cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_dir / Path(src_path).name)


# ── Statistics ────────────────────────────────────────────────────────────────

def print_dataset_stats(train, val, test, classes):
    """Print per-class image counts and imbalance ratio."""
    label_counter = Counter(s[1] for s in train)
    counts = [label_counter.get(c, 0) for c in classes]
    max_c  = max(counts)
    min_c  = min(counts) if min(counts) > 0 else 1

    print("\n── Dataset Statistics ──────────────────────────")
    print(f"  Train : {len(train):,} images")
    print(f"  Val   : {len(val):,} images")
    print(f"  Test  : {len(test):,} images")
    print(f"  Total : {len(train)+len(val)+len(test):,} images")
    print(f"  Classes: {len(classes)}")
    print(f"\n  Class imbalance ratio (max/min): {max_c/min_c:.2f}x")
    print("\n  Per-class (train) sample counts:")
    for cls, cnt in zip(classes, counts):
        bar = "█" * (cnt // 50)
        print(f"    {cls:<45} {cnt:>5}  {bar}")


def save_class_names(classes, output_path: str):
    """Save {index: class_name} mapping to JSON."""
    mapping = {i: cls for i, cls in enumerate(classes)}
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  ✓ class_names.json saved → {output_path}")


# ── Sample Augmentation Visualization ─────────────────────────────────────────

def save_sample_augmentations(samples, classes, output_dir: str, n: int = 5, seed: int = 42):
    """Save n augmented samples to output_dir."""
    random.seed(seed)
    chosen = random.sample(samples, min(n, len(samples)))
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    for i, (img_path, cls_name) in enumerate(chosen):
        original = Image.open(img_path).convert("RGB")
        aug_t    = TRAIN_TRANSFORM(original)   # tensor
        # denormalize for display
        mean_t = torch.tensor(MEAN).view(3, 1, 1)
        std_t  = torch.tensor(STD).view(3, 1, 1)
        aug_display = (aug_t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[1].imshow(aug_display)
        axes[1].set_title("Augmented")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(cls_name, fontsize=10)
        plt.tight_layout()
        plt.savefig(out / f"sample_aug_{i:02d}_{cls_name.replace('/', '_')}.png", dpi=100)
        plt.close()

    print(f"  ✓ {n} sample augmentations saved → {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PlantVillage preprocessing")
    parser.add_argument("--raw_dir",    default="data/raw/plantvillage")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--aug_dir",    default="notebooks/sample_augmentations")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print(f"\n[1/5] Collecting images from: {args.raw_dir}")
    samples, classes = collect_image_paths(args.raw_dir)
    print(f"      Found {len(samples):,} images across {len(classes)} classes")

    if len(samples) == 0:
        print(f"\n❌ No images found in '{args.raw_dir}'.")
        print("   Please download PlantVillage first:")
        print("   → python src/data/download_plantvillage.py")
        print("   Then re-run this script.\n")
        import sys; sys.exit(1)

    print(f"\n[2/5] Stratified split (80/10/10, seed={args.seed})")
    train, val, test = stratified_split(samples, seed=args.seed)

    print(f"\n[3/5] Copying files to {args.output_dir}")
    copy_split_to_output(train, "train", args.output_dir)
    copy_split_to_output(val,   "val",   args.output_dir)
    copy_split_to_output(test,  "test",  args.output_dir)

    print(f"\n[4/5] Saving class_names.json")
    save_class_names(classes, str(Path(args.output_dir) / "class_names.json"))

    print(f"\n[5/5] Saving sample augmentations")
    save_sample_augmentations(train[:50], classes, args.aug_dir, n=5, seed=args.seed)

    print_dataset_stats(train, val, test, classes)
    print("\n✓ Preprocessing complete!\n")


if __name__ == "__main__":
    main()
