"""
Evaluation script — runs full test-set evaluation, per-class accuracy,
confusion matrix, and GradCAM samples. Outputs metrics JSON.
Stage 5 — Evaluation & Testing
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
)
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.model    import EfficientNetB4Classifier
from src.vision.gradcam  import GradCAM, batch_visualize
from src.vision.preprocess import VAL_TRANSFORM


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    return all_preds, all_labels, all_probs


def compute_metrics(preds, labels, probs, class_names, output_dir):
    """Compute and save all evaluation metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(labels, preds, average="weighted", zero_division=0)

    # AUC (one-vs-rest, macro)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            per_class[cls] = float(accuracy_score(labels[mask], preds[mask]))

    results = {
        "accuracy":        round(float(acc),    4),
        "f1_macro":        round(float(f1_mac), 4),
        "f1_weighted":     round(float(f1_wt),  4),
        "auc_macro_ovr":   round(float(auc), 4) if auc else None,
        "per_class_accuracy": per_class,
        "num_samples":     int(len(labels)),
        "num_classes":     len(class_names),
    }

    # Save metrics JSON
    metrics_path = output_dir / "eval_results.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 (macro) : {f1_mac:.4f}")
    print(f"  F1 (wgtd)  : {f1_wt:.4f}")
    if auc:
        print(f"  AUC (macro): {auc:.4f}")
    print(f"\n  ✓ Metrics saved → {metrics_path}")

    return results


def plot_confusion_matrix(preds, labels, class_names, output_dir, top_n=20):
    """Plot and save confusion matrix (top N most frequent classes)."""
    output_dir = Path(output_dir)

    # Only show top_n classes for readability
    counts   = np.bincount(labels, minlength=len(class_names))
    top_idx  = np.argsort(counts)[-top_n:][::-1]
    mask     = np.isin(labels, top_idx)
    sub_pred = preds[mask]
    sub_lbl  = labels[mask]

    # Remap to 0..top_n-1
    idx_map  = {old: new for new, old in enumerate(top_idx)}
    sub_pred = np.array([idx_map.get(p, -1) for p in sub_pred])
    sub_lbl  = np.array([idx_map[l] for l in sub_lbl])
    valid    = sub_pred >= 0
    sub_pred, sub_lbl = sub_pred[valid], sub_lbl[valid]

    cm       = confusion_matrix(sub_lbl, sub_pred, labels=list(range(len(top_idx))))
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    names = [class_names[i].split("___")[-1][:20] for i in top_idx]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"Confusion Matrix (top {top_n} classes, normalized)", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {cm_path}")
    return cm_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/vision/efficientnet_b4_best.pt")
    parser.add_argument("--data_dir",    default="data/processed/test")
    parser.add_argument("--class_names", default="data/processed/class_names.json")
    parser.add_argument("--output",      default="models/vision")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--gradcam_n",   type=int, default=10,
                        help="Number of Grad-CAM samples to visualize")
    parser.add_argument("--wandb",       action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")

    # ── Class names ───────────────────────────────────────────────────────────
    with open(args.class_names) as f:
        mapping     = json.load(f)
        class_names = [mapping[str(i)] for i in range(len(mapping))]

    # ── Data ──────────────────────────────────────────────────────────────────
    test_ds     = datasets.ImageFolder(args.data_dir, transform=VAL_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EfficientNetB4Classifier(num_classes=len(class_names)).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds, labels, probs = run_evaluation(model, test_loader, device, len(class_names))
    results = compute_metrics(preds, labels, probs, class_names, args.output)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = plot_confusion_matrix(preds, labels, class_names, args.output)

    # ── Grad-CAM samples ──────────────────────────────────────────────────────
    if args.gradcam_n > 0:
        import random
        random.seed(42)
        sample_paths = random.sample(test_ds.samples, min(args.gradcam_n, len(test_ds)))
        samples = [(p, class_names[l]) for p, l in sample_paths]
        gradcam_dir = str(Path(args.output) / "gradcam_samples")
        batch_visualize(
            model=model,
            target_layer=model.get_target_layer(),
            samples=samples,
            transform=VAL_TRANSFORM,
            output_dir=gradcam_dir,
            class_names=class_names,
            device=device,
        )

    # ── W&B logging ───────────────────────────────────────────────────────────
    if args.wandb:
        run = wandb.init(project="crop-disease-advisor", job_type="evaluation")
        wandb.log({
            "test/accuracy":   results["accuracy"],
            "test/f1_macro":   results["f1_macro"],
            "test/f1_weighted":results["f1_weighted"],
            "test/auc":        results.get("auc_macro_ovr") or 0,
        })
        wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})
        wandb.finish()

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
