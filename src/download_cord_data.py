from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick downloader for CORD data -> GCN CSV")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--dataset-id", default="naver-clova-ix/cord-v2")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    out_csv = project_dir / "data" / f"cord_{args.split}_nodes.csv"

    cmd = [
        sys.executable,
        "pipeline_runner.py",
        "convert_hf_cord_to_csv",
        "--dataset-id",
        args.dataset_id,
        "--split",
        args.split,
        "--output-csv",
        str(out_csv),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    subprocess.run(cmd, check=True, cwd=project_dir)


if __name__ == "__main__":
    main()
