"""
EfficientNet-B4 training script with gradual unfreezing, W&B logging,
mixed precision, and early stopping.
Prompt 1.5 — Training Script with Gradual Unfreezing
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.model import EfficientNetB4Classifier, model_summary
from src.vision.preprocess import TRAIN_TRANSFORM, VAL_TRANSFORM


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Unfreezing ────────────────────────────────────────────────────────────────

def set_phase1(model: EfficientNetB4Classifier, lr: float, optimizer):
    """Freeze backbone, train classifier head only."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for g in optimizer.param_groups:
        g["lr"] = lr
    print(f"  Phase 1: classifier head only  | lr={lr}")


def set_phase2(model: EfficientNetB4Classifier, lr: float, optimizer):
    """Unfreeze last 2 MBConv blocks."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    # blocks[-1] and blocks[-2]
    for block in model.backbone.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for g in optimizer.param_groups:
        g["lr"] = lr
    print(f"  Phase 2: last 2 MBConv blocks  | lr={lr}")


def set_phase3(model: EfficientNetB4Classifier, lr: float, optimizer):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    for g in optimizer.param_groups:
        g["lr"] = lr
    print(f"  Phase 3: full model            | lr={lr}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        total_acc  += accuracy(logits.detach(), labels) * imgs.size(0)
        n          += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_acc  += accuracy(logits, labels) * imgs.size(0)
        n          += imgs.size(0)
    return total_loss / n, total_acc / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg.get("wandb_project", "crop-disease-advisor"),
        config=cfg,
        name=f"efficientnet_b4_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = cfg["data_root"]
    train_ds  = datasets.ImageFolder(f"{data_root}/train", transform=TRAIN_TRANSFORM)
    val_ds    = datasets.ImageFolder(f"{data_root}/val",   transform=VAL_TRANSFORM)
    test_ds   = datasets.ImageFolder(f"{data_root}/test",  transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = EfficientNetB4Classifier(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    model_summary(model, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr_phase1"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"])
    scaler    = GradScaler()

    # Checkpoint dir
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["best_model_name"]

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    p1_end = cfg["unfreeze_schedule"]["phase1_end"]
    p2_end = cfg["unfreeze_schedule"]["phase2_end"]

    set_phase1(model, cfg["lr_phase1"], optimizer)

    for epoch in range(cfg["epochs"]):
        # Unfreezing transitions
        if epoch == p1_end:
            set_phase2(model, cfg["lr_phase2"], optimizer)
        elif epoch == p2_end:
            set_phase3(model, cfg["lr_phase3"], optimizer)

        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        # Best checkpoint
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg["early_stop_patience"]:
                print(f"\n⚠ Early stopping at epoch {epoch+1} (patience={cfg['early_stop_patience']})")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Final Test Evaluation ──────────────────────")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}")
    wandb.log({"test_acc": test_acc, "test_loss": test_loss})

    # Per-class accuracy
    model.eval()
    class_correct = torch.zeros(cfg["num_classes"])
    class_total   = torch.zeros(cfg["num_classes"])
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            for i in range(len(labels)):
                c = labels[i].item()
                class_correct[c] += (preds[i] == labels[i]).item()
                class_total[c]   += 1

    class_names = train_ds.classes
    print("\n  Per-class Test Accuracy:")
    for i, cls in enumerate(class_names):
        acc_c = class_correct[i] / max(class_total[i], 1)
        print(f"    {cls:<45} {acc_c:.3f}")

    wandb.finish()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
