"""
Model registration pipeline — pushes evaluated model to W&B Model Registry
and HuggingFace Hub with full metadata tagging.
Stage 4 — Model Versioning & Registry
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import wandb
from huggingface_hub import HfApi


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_eval_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_checkpoint_meta(ckpt_path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    return {
        "epoch":   ckpt.get("epoch", "unknown"),
        "val_acc": ckpt.get("val_acc", None),
    }


# ── W&B Model Registry ────────────────────────────────────────────────────────

def register_to_wandb(
    ckpt_path:    str,
    eval_path:    str,
    project:      str,
    model_name:   str,
    stage:        str = "staging",   # "staging" or "production"
    tags:         list = None,
):
    """Log model artifact to W&B and promote to registry."""
    eval_res  = load_eval_results(eval_path)
    ckpt_meta = load_checkpoint_meta(ckpt_path)

    run = wandb.init(
        project=project,
        job_type="model-registration",
        name=f"register_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=eval_res,
    )

    # Create and log artifact
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="EfficientNet-B4 plant disease classifier",
        metadata={
            "epoch":       ckpt_meta["epoch"],
            "val_acc":     ckpt_meta["val_acc"],
            "test_acc":    eval_res.get("accuracy"),
            "f1_macro":    eval_res.get("f1_macro"),
            "f1_weighted": eval_res.get("f1_weighted"),
            "auc":         eval_res.get("auc_macro_ovr"),
            "num_classes": eval_res.get("num_classes"),
            "registered":  datetime.now().isoformat(),
        },
    )
    artifact.add_file(ckpt_path, name="efficientnet_b4_best.pt")

    # Attach eval results
    if Path(eval_path).exists():
        artifact.add_file(eval_path, name="eval_results.json")

    cm_path = str(Path(ckpt_path).parent / "confusion_matrix.png")
    if Path(cm_path).exists():
        artifact.add_file(cm_path, name="confusion_matrix.png")

    run.log_artifact(artifact, aliases=[stage, "latest"])

    # Log summary metrics
    wandb.run.summary.update({
        "test_accuracy": eval_res.get("accuracy"),
        "f1_macro":      eval_res.get("f1_macro"),
        "stage":         stage,
    })

    wandb.finish()
    print(f"  ✓ Model logged to W&B artifact registry as '{model_name}:{stage}'")


# ── HuggingFace Hub ───────────────────────────────────────────────────────────

def register_to_hf(
    ckpt_path:    str,
    eval_path:    str,
    class_names_path: str,
    username:     str,
    token:        str,
    repo_suffix:  str = "crop-disease-efficientnet-b4",
    commit_msg:   str = None,
):
    """Upload model checkpoint and metadata to HuggingFace Hub."""
    api     = HfApi()
    repo_id = f"{username}/{repo_suffix}"

    # Ensure repo exists
    api.create_repo(repo_id=repo_id, repo_type="model",
                    exist_ok=True, token=token)

    eval_res = load_eval_results(eval_path)
    msg = commit_msg or (
        f"Update checkpoint | test_acc={eval_res.get('accuracy',0):.4f} "
        f"f1={eval_res.get('f1_macro',0):.4f}"
    )

    # Upload checkpoint
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo="efficientnet_b4_best.pt",
        repo_id=repo_id, repo_type="model",
        token=token, commit_message=msg,
    )

    # Upload class names
    if Path(class_names_path).exists():
        api.upload_file(
            path_or_fileobj=class_names_path,
            path_in_repo="class_names.json",
            repo_id=repo_id, repo_type="model",
            token=token,
        )

    # Upload eval results
    if Path(eval_path).exists():
        api.upload_file(
            path_or_fileobj=eval_path,
            path_in_repo="eval_results.json",
            repo_id=repo_id, repo_type="model",
            token=token,
        )

    print(f"  ✓ Model uploaded to https://huggingface.co/{repo_id}")


# ── Promotion Check ───────────────────────────────────────────────────────────

def check_promotion_gate(eval_path: str, min_accuracy: float = 0.85) -> bool:
    """Return True only if model meets the minimum accuracy threshold."""
    eval_res = load_eval_results(eval_path)
    acc      = eval_res.get("accuracy", 0)
    if acc >= min_accuracy:
        print(f"  ✓ Promotion gate PASSED — accuracy={acc:.4f} ≥ {min_accuracy}")
        return True
    else:
        print(f"  ❌ Promotion gate FAILED — accuracy={acc:.4f} < {min_accuracy}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Register trained model to W&B + HuggingFace")
    parser.add_argument("--checkpoint",   default="models/vision/efficientnet_b4_best.pt")
    parser.add_argument("--eval_path",    default="models/vision/eval_results.json")
    parser.add_argument("--class_names",  default="data/processed/class_names.json")
    parser.add_argument("--project",      default="crop-disease-advisor")
    parser.add_argument("--model_name",   default="efficientnet-b4-plant-disease")
    parser.add_argument("--stage",        default="staging",
                        choices=["staging", "production"])
    parser.add_argument("--hf_username",  default=os.getenv("HF_USERNAME", ""))
    parser.add_argument("--hf_token",     default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--min_accuracy", type=float, default=0.85)
    parser.add_argument("--skip_gate",    action="store_true",
                        help="Skip promotion accuracy gate")
    args = parser.parse_args()

    print("\n── Model Registration Pipeline ─────────────────────")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Stage      : {args.stage}")

    # ── Promotion gate ────────────────────────────────────────────────────────
    if not args.skip_gate:
        if not check_promotion_gate(args.eval_path, args.min_accuracy):
            print("\n⚠ Registration aborted. Improve model before promoting.")
            return

    # ── W&B ──────────────────────────────────────────────────────────────────
    print("\n[1/2] Registering to W&B Model Registry...")
    register_to_wandb(
        ckpt_path=args.checkpoint,
        eval_path=args.eval_path,
        project=args.project,
        model_name=args.model_name,
        stage=args.stage,
    )

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    if args.hf_username and args.hf_token:
        print("\n[2/2] Uploading to HuggingFace Hub...")
        register_to_hf(
            ckpt_path=args.checkpoint,
            eval_path=args.eval_path,
            class_names_path=args.class_names,
            username=args.hf_username,
            token=args.hf_token,
            commit_msg=f"[{args.stage}] test_acc={load_eval_results(args.eval_path).get('accuracy',0):.4f}",
        )
    else:
        print("\n[2/2] Skipping HuggingFace upload (no HF_USERNAME/HF_TOKEN).")

    print("\n✅ Model registration complete!")


if __name__ == "__main__":
    main()
