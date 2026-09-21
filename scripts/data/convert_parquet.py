"""
Convert HuggingFace Parquet dataset files to ImageFolder structure.

Use this if you manually downloaded the Parquet files from:
  https://huggingface.co/datasets/nateraw/plant-disease/tree/main/data

Usage:
    python3 src/data/convert_parquet.py --parquet_dir ~/Downloads/plant-disease
    python3 src/data/convert_parquet.py --parquet_glob "data/raw/*.parquet"
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from io import BytesIO


def convert_parquet_to_imagefolder(parquet_files: list, output_dir: str):
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not installed. Run: pip3 install pandas pyarrow")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("❌ Pillow not installed. Run: pip3 install Pillow")
        sys.exit(1)

    output_path = Path(output_dir)
    total_saved = 0

    for pq_file in parquet_files:
        print(f"  Reading: {pq_file}")
        df = pd.read_parquet(pq_file)
        print(f"  Columns: {list(df.columns)}  Rows: {len(df)}")

        # Find image and label columns
        img_col   = next((c for c in df.columns if "image" in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns if c in ["label","labels","class","disease"]),
                         [c for c in df.columns if c != img_col][0])

        # Class names from pandas CategoricalDtype or unique values
        if hasattr(df[label_col], "cat"):
            class_names = list(df[label_col].cat.categories)
        else:
            unique_labels = sorted(df[label_col].unique())
            class_names   = [str(l) for l in unique_labels]

        print(f"  Classes: {len(class_names)}")

        for i, row in df.iterrows():
            label = row[label_col]
            cls_name = class_names[label] if isinstance(label, int) else str(label)
            cls_dir  = output_path / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            img_data = row[img_col]
            img_path = cls_dir / f"{total_saved:07d}.jpg"

            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
            elif isinstance(img_data, bytes):
                img = Image.open(BytesIO(img_data)).convert("RGB")
            else:
                img = img_data  # assume PIL image

            img.save(img_path, format="JPEG", quality=95)
            total_saved += 1

            if (total_saved % 500) == 0:
                print(f"  Progress: {total_saved:,} images saved...", end="\r", flush=True)

    print(f"\n✅ Converted {total_saved:,} images → {output_path}")

    # Quick verify
    classes = [d for d in output_path.iterdir() if d.is_dir()]
    print(f"   Classes : {len(classes)}")
    print(f"\n  Now run:")
    print(f"   python3 src/vision/preprocess.py")


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet → ImageFolder")
    parser.add_argument("--parquet_dir",  default=None, help="Directory containing .parquet files")
    parser.add_argument("--parquet_glob", default=None, help="Glob pattern e.g. 'downloads/*.parquet'")
    parser.add_argument("--output",       default="data/raw/plantvillage")
    args = parser.parse_args()

    if args.parquet_dir:
        files = sorted(glob.glob(str(Path(args.parquet_dir) / "*.parquet")))
    elif args.parquet_glob:
        files = sorted(glob.glob(args.parquet_glob))
    else:
        parser.print_help()
        print("\n❌ Provide --parquet_dir or --parquet_glob")
        sys.exit(1)

    if not files:
        print(f"❌ No .parquet files found.")
        sys.exit(1)

    print(f"Found {len(files)} Parquet file(s):")
    for f in files:
        print(f"  {f}")
    print()

    convert_parquet_to_imagefolder(files, args.output)


if __name__ == "__main__":
    main()
