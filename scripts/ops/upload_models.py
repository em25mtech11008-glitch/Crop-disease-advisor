"""
Upload trained models to HuggingFace Hub.
Prompt 2.8 — Upload Models Script
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_models(username: str, token: str):
    api = HfApi()

    # ── Vision Model ──────────────────────────────────────────────────────────
    vision_repo = f"{username}/crop-disease-efficientnet-b4"
    print(f"\n[1/2] Uploading EfficientNet-B4 → {vision_repo}")

    create_repo(vision_repo, repo_type="model", exist_ok=True, token=token)

    vision_ckpt = Path("models/vision/efficientnet_b4_best.pt")
    if not vision_ckpt.exists():
        print(f"  ❌ Checkpoint not found: {vision_ckpt}")
    else:
        api.upload_file(
            path_or_fileobj=str(vision_ckpt),
            path_in_repo="efficientnet_b4_best.pt",
            repo_id=vision_repo,
            repo_type="model",
            token=token,
            commit_message="Upload EfficientNet-B4 Phase 1 best checkpoint",
        )
        # Upload class names
        class_names = Path("data/processed/class_names.json")
        if class_names.exists():
            api.upload_file(
                path_or_fileobj=str(class_names),
                path_in_repo="class_names.json",
                repo_id=vision_repo,
                repo_type="model",
                token=token,
            )
        print(f"  ✓  Vision model uploaded")
        print(f"     → https://huggingface.co/{vision_repo}")

    # ── LLM Adapter ──────────────────────────────────────────────────────────
    llm_repo = f"{username}/crop-disease-llama3-qlora"
    print(f"\n[2/2] Uploading LLaMA 3 QLoRA adapter → {llm_repo}")

    create_repo(llm_repo, repo_type="model", exist_ok=True, token=token)

    adapter_dir = Path("models/llm/llama3_qlora_adapter")
    if not adapter_dir.exists():
        print(f"  ❌ Adapter directory not found: {adapter_dir}")
    else:
        api.upload_folder(
            folder_path=str(adapter_dir),
            path_in_repo=".",
            repo_id=llm_repo,
            repo_type="model",
            token=token,
            commit_message="Upload LLaMA 3 8B QLoRA adapter — crop disease advisor",
        )
        print(f"  ✓  LLM adapter uploaded")
        print(f"     → https://huggingface.co/{llm_repo}")

    print("\n✅ All models uploaded to HuggingFace Hub!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument("--token",    default=os.getenv("HF_TOKEN",""), help="HF API token")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("HF_TOKEN not set. Pass --token or set HF_TOKEN env var.")

    upload_models(args.username, args.token)
