#!/usr/bin/env python3
"""Robust downloader for flwrlabs/cinic10 from HF mirror."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download flwrlabs/cinic10 with acceleration, resume and retries."
    )
    parser.add_argument(
        "--repo-id",
        default="flwrlabs/cinic10",
        help="Dataset repo id on Hugging Face Hub (default: flwrlabs/cinic10).",
    )
    parser.add_argument(
        "--output-dir",
        default="./cinic10_hf",
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./.hf_cache",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Hub endpoint (default: https://hf-mirror.com).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Maximum retry attempts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(4, min(16, (os.cpu_count() or 4) * 2)),
        help="Concurrent workers for snapshot download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Download timeout (seconds) for each file request.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional for public dataset).",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=200 * 1024 * 1024,
        help="Minimum expected total bytes for a successful download check.",
    )
    return parser.parse_args()


def setup_env(args: argparse.Namespace) -> None:
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HOME"] = str(Path(args.cache_dir).resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir).resolve())
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

    # Accelerate transfer if hf_transfer is available.
    try:
        __import__("hf_transfer")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[INFO] hf_transfer detected, high-speed transfer enabled.")
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("[INFO] hf_transfer not installed, using standard transfer.")


def human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num}B"


def validate_download(output_dir: Path, min_bytes: int) -> bool:
    if not output_dir.exists():
        return False

    files = [p for p in output_dir.rglob("*") if p.is_file()]
    if not files:
        return False

    parquet_files = [p for p in files if p.suffix.lower() == ".parquet"]
    total_bytes = sum(p.stat().st_size for p in files)

    print(
        f"[INFO] Validation: files={len(files)}, parquet={len(parquet_files)}, "
        f"size={human_bytes(total_bytes)}"
    )
    return len(parquet_files) > 0 and total_bytes >= min_bytes


def snapshot_download_with_retry(args: argparse.Namespace) -> bool:
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.retries + 1):
        print(f"[INFO] snapshot_download attempt {attempt}/{args.retries} ...")
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=args.max_workers,
                token=args.token,
            )
            if validate_download(output_dir, args.min_bytes):
                print("[INFO] snapshot_download finished and validated.")
                return True
            print("[WARN] Download exists but validation is not passed, retrying.")
        except Exception as err:
            print(f"[WARN] snapshot_download failed on attempt {attempt}: {err}")

        if attempt < args.retries:
            wait_seconds = min(90, 2 ** attempt + random.uniform(0, 1.5))
            print(f"[INFO] Sleeping {wait_seconds:.1f}s before next retry...")
            time.sleep(wait_seconds)

    return False


def datasets_fallback_with_retry(args: argparse.Namespace) -> bool:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max(3, args.retries // 2) + 1):
        print(f"[INFO] datasets fallback attempt {attempt} ...")
        try:
            from datasets import load_dataset

            ds = load_dataset(
                args.repo_id,
                cache_dir=str(Path(args.cache_dir).resolve()),
            )
            marker = output_dir / "dataset_dict_summary.txt"
            with marker.open("w", encoding="utf-8") as f:
                for split_name, split_ds in ds.items():
                    f.write(f"{split_name}\t{len(split_ds)}\n")
            print(f"[INFO] Fallback succeeded. Split summary saved to: {marker}")
            return True
        except Exception as err:
            print(f"[WARN] datasets fallback failed on attempt {attempt}: {err}")
            if attempt < max(3, args.retries // 2):
                wait_seconds = min(60, 3 * attempt + random.uniform(0, 1.0))
                print(f"[INFO] Sleeping {wait_seconds:.1f}s before retry...")
                time.sleep(wait_seconds)

    return False


def main() -> int:
    args = parse_args()
    setup_env(args)

    print(f"[INFO] Repo: {args.repo_id}")
    print(f"[INFO] Output dir: {Path(args.output_dir).resolve()}")
    print(f"[INFO] Cache dir:  {Path(args.cache_dir).resolve()}")
    print(f"[INFO] Endpoint:   {args.endpoint}")
    print(f"[INFO] Workers:    {args.max_workers}")

    try:
        from huggingface_hub import __version__ as hub_version

        print(f"[INFO] huggingface_hub version: {hub_version}")
    except Exception:
        print("[ERROR] huggingface_hub is not installed. Run:")
        print("        pip install -U huggingface_hub datasets")
        return 1

    ok = snapshot_download_with_retry(args)
    if ok:
        print("[SUCCESS] CINIC10 downloaded successfully.")
        return 0

    print("[WARN] snapshot_download failed after all retries. Trying datasets fallback.")
    ok = datasets_fallback_with_retry(args)
    if ok:
        print("[SUCCESS] CINIC10 downloaded via fallback method.")
        return 0

    print("[ERROR] All download methods failed.")
    print("[TIP] Try lowering --max-workers (e.g. 4) or increasing --retries.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
