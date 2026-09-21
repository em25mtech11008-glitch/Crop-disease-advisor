"""
Download PlantVillage dataset — multiple fallback methods.

Usage:
    python3 src/data/download_plantvillage.py                  # auto (tries all)
    python3 src/data/download_plantvillage.py --method hf      # HuggingFace datasets
    python3 src/data/download_plantvillage.py --method kaggle  # Kaggle API
    python3 src/data/download_plantvillage.py --method wget    # direct wget (fastest)
"""

import os, sys, shutil, argparse, zipfile, subprocess
from pathlib import Path


# ── Correct HuggingFace dataset IDs (verified) ────────────────────────────────
HF_CANDIDATES = [
    "aymen31/PlantVillage",                 # 38 classes, public
    "dpdl-benchmark/plant_village",         # 38 classes
    "imadhajaz/plant_village",              # mirrors
]

# ─── Method 1: HuggingFace datasets library ───────────────────────────────────

def download_via_huggingface(output_dir: str) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ `datasets` not installed. Run: pip3 install datasets")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for repo_id in HF_CANDIDATES:
        print(f"   Trying: {repo_id} ...")
        try:
            ds = load_dataset(repo_id, split="train")
        except Exception as e:
            print(f"   ✗ {e}")
            continue

        print(f"   ✓ Loaded {len(ds)} samples from {repo_id}")

        # Find label column
        label_col = next(
            (c for c in ["label", "labels", "class", "disease_class"]
             if c in ds.features), None
        )
        if label_col is None:
            label_col = [c for c in ds.features if c != "image"][0]

        label_feat = ds.features[label_col]
        class_names = label_feat.names if hasattr(label_feat, "names") \
                      else [str(i) for i in range(max(ds[label_col]) + 1)]

        print(f"   Classes : {len(class_names)}")
        print(f"   Saving  → {output_path}\n")

        for i, sample in enumerate(ds):
            cls_name = class_names[sample[label_col]]
            cls_dir  = output_path / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            img_path = cls_dir / f"{i:06d}.jpg"
            if not img_path.exists():
                sample["image"].convert("RGB").save(img_path, format="JPEG", quality=95)
            if (i + 1) % 1000 == 0:
                print(f"   {i+1:,}/{len(ds):,} saved...", end="\r", flush=True)

        print(f"\n✅ {len(ds):,} images saved to {output_path}")
        return True

    return False


# ─── Method 2: HuggingFace snapshot_download (dataset repo) ──────────────────

def download_via_hf_snapshot(output_dir: str, token: str = None) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ `huggingface_hub` not installed.")
        return False

    # This repo has the full PlantVillage in ImageFolder structure
    SNAPSHOT_REPOS = [
        "aymen31/PlantVillage",
        "dpdl-benchmark/plant_village",
    ]

    output_path = Path(output_dir)

    for repo_id in SNAPSHOT_REPOS:
        print(f"   Trying snapshot: {repo_id} ...")
        try:
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=token or os.getenv("HF_TOKEN"),
                local_dir=str(output_path),
                ignore_patterns=["*.parquet", "*.json", "*.md", ".gitattributes"],
            )
            img_count = sum(1 for _ in output_path.rglob("*.jpg")) + \
                        sum(1 for _ in output_path.rglob("*.png"))
            if img_count > 0:
                print(f"✅ Snapshot downloaded: {img_count:,} images → {output_path}")
                return True
        except Exception as e:
            print(f"   ✗ {e}")

    return False


# ─── Method 3: wget / curl direct download ───────────────────────────────────

DIRECT_URLS = [
    # PlantVillage color images via public mirror (Mendeley Data)
    # Add any publicly accessible zip URL for your institution here
    # Example (replace with an actual accessible URL):
    ("wget", "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tywbtsjrjv-1.zip"),
]

def download_via_wget(output_dir: str) -> bool:
    """
    Download PlantVillage zip via wget/curl and extract.
    Update DIRECT_URLS above with a URL accessible from your HPC.
    """
    output_path = Path(output_dir)
    parent      = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path    = parent / "plantvillage.zip"

    for tool, url in DIRECT_URLS:
        print(f"   Trying {tool}: {url[:70]}...")
        if not shutil.which(tool):
            print(f"   ✗ Executable '{tool}' not found.")
            continue
            
        if tool == "wget":
            ret = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(zip_path), url],
                check=False
            ).returncode
        else:
            ret = subprocess.run(
                ["curl", "-L", "-o", str(zip_path), url],
                check=False
            ).returncode

        if ret == 0 and zip_path.exists() and zip_path.stat().st_size > 1_000_000:
            print(f"   ✓ Downloaded {zip_path.stat().st_size // 1024**2} MB")
            print(f"   Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(parent))
            zip_path.unlink()
            # Find the extracted train folder and move to expected location
            for candidate in parent.rglob("train"):
                if candidate.is_dir():
                    if output_path.exists():
                        shutil.rmtree(output_path)
                    shutil.move(str(candidate), str(output_path))
                    print(f"✅ Extracted to {output_path}")
                    return True
        else:
            print(f"   ✗ Download failed (exit code {ret})")

    return False


# ─── Method 4: Kaggle API ─────────────────────────────────────────────────────

def download_via_kaggle(output_dir: str) -> bool:
    """Requires KAGGLE_USERNAME and KAGGLE_KEY env vars or ~/.kaggle/kaggle.json"""
    output_path = Path(output_dir)
    parent      = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and not os.getenv("KAGGLE_KEY"):
        print("❌ Kaggle credentials not found.")
        print("   Set KAGGLE_USERNAME + KAGGLE_KEY env vars, or create ~/.kaggle/kaggle.json")
        print("   Get your key from: https://www.kaggle.com/settings → API → Create Token")
        return False

    if not shutil.which("kaggle"):
        print("❌ `kaggle` executable not found.")
        print("   Run: pip install kaggle")
        return False

    print("   Downloading from Kaggle: vipoooool/new-plant-diseases-dataset")
    ret = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "vipoooool/new-plant-diseases-dataset",
        "-p", str(parent), "--unzip"
    ], check=False).returncode

    if ret != 0:
        print("❌ Kaggle download failed.")
        return False

    # Fix folder structure — find the train/ dir
    for candidate in sorted(parent.rglob("train"), key=lambda p: len(p.parts)):
        if candidate.is_dir() and any(candidate.iterdir()):
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.move(str(candidate), str(output_path))
            print(f"✅ Dataset moved to {output_path}")
            return True

    print("⚠  Could not locate extracted train/ folder. Check data/raw/ manually.")
    return False


# ─── Verify ──────────────────────────────────────────────────────────────────

def verify_download(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n❌ Directory not found: {data_dir}")
        return False

    classes   = sorted([d for d in data_path.iterdir() if d.is_dir()])
    total_img = sum(
        len([f for f in c.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        for c in classes
    )

    print(f"\n── Verification ─────────────────────────────────")
    print(f"  Directory : {data_dir}")
    print(f"  Classes   : {len(classes)}")
    print(f"  Images    : {total_img:,}")

    if len(classes) > 0:
        print(f"\n  Sample classes:")
        for cls in classes[:5]:
            n = len(list(cls.iterdir()))
            print(f"    {cls.name:<45} {n} images")
        if len(classes) > 5:
            print(f"    ... and {len(classes)-5} more")

    if total_img > 1000:
        print(f"\n  ✅ Data ready! Now run:")
        print(f"     python3 src/vision/preprocess.py")
        return True
    else:
        print(f"\n  ❌ Too few images ({total_img}). Download may have failed.")
        return False


# ─── Manual instructions ─────────────────────────────────────────────────────

def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Manual Download Instructions (All Methods Failed)     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Option A — Kaggle (recommended):                                ║
║    1. Go to: https://www.kaggle.com/datasets/vipoooool/          ║
║             new-plant-diseases-dataset                           ║
║    2. Download ZIP and upload to HPC via scp:                    ║
║       scp plantvillage.zip m25csa013@hpc.iitj.ac.in:~/          ║
║    3. Extract:                                                   ║
║       mkdir -p data/raw/plantvillage                             ║
║       unzip ~/plantvillage.zip -d /tmp/pv                       ║
║       mv /tmp/pv/*/train/* data/raw/plantvillage/               ║
║                                                                  ║
║  Option B — HuggingFace web download:                            ║
║    1. Go to: https://huggingface.co/datasets/                    ║
║             nateraw/plant-disease                                ║
║    2. Download the Parquet files and convert:                    ║
║       python3 src/data/convert_parquet.py                        ║
║                                                                  ║
║  Option C — If dataset already on HPC shared storage:           ║
║    Ask your system admin for the path, then symlink:             ║
║    ln -s /path/to/plantvillage data/raw/plantvillage             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset")
    parser.add_argument("--method",  default="auto",
                        choices=["auto","hf","snapshot","kaggle","wget"])
    parser.add_argument("--output",  default="data/raw/plantvillage")
    parser.add_argument("--token",   default=os.getenv("HF_TOKEN",""))
    args = parser.parse_args()

    print(f"\n{'═'*50}")
    print(f"  PlantVillage Downloader  |  method={args.method}")
    print(f"{'═'*50}\n")

    success = False

    if args.method in ("auto", "hf"):
        print("[1] Trying HuggingFace datasets library...")
        success = download_via_huggingface(args.output)

    if not success and args.method in ("auto", "snapshot"):
        print("\n[2] Trying HuggingFace snapshot_download...")
        success = download_via_hf_snapshot(args.output, args.token)

    if not success and args.method in ("auto", "kaggle"):
        print("\n[3] Trying Kaggle API...")
        success = download_via_kaggle(args.output)

    if not success and args.method in ("auto", "wget"):
        print("\n[4] Trying direct wget...")
        success = download_via_wget(args.output)

    if not success:
        print("\n❌ All automatic download methods failed.")
        print_manual_instructions()
        sys.exit(1)

    verify_download(args.output)


if __name__ == "__main__":
    main()
