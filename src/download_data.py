from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic HF dataset downloader -> GCN CSV")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--doc-id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--bbox-field", default="bbox")
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--label-map", default=None, help="JSON string or path to JSON file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--streaming", type=int, default=1, help="1=low RAM mode, 0=normal mode")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_csv = args.output_csv or str(project_dir / "data" / f"{args.split}_nodes.csv")

    is_cord = "cord" in args.dataset_id.lower()

    if is_cord:
        # CORD has custom schema (ground_truth JSON), use dedicated converter.
        cmd = [
            sys.executable,
            "pipeline_runner.py",
            "convert_hf_cord_to_csv",
            "--dataset-id",
            args.dataset_id,
            "--split",
            args.split,
            "--output-csv",
            output_csv,
            "--streaming",
            str(args.streaming),
        ]
    else:
        # Generic converter for datasets that expose text/label/bbox style fields.
        cmd = [
            sys.executable,
            "pipeline_runner.py",
            "convert_hf_to_gcn_csv",
            "--dataset-id",
            args.dataset_id,
            "--split",
            args.split,
            "--output-csv",
            output_csv,
            "--doc-id-field",
            args.doc_id_field,
            "--text-field",
            args.text_field,
            "--label-field",
            args.label_field,
            "--bbox-field",
            args.bbox_field,
        ]

    if not is_cord:
        if args.score_field:
            cmd.extend(["--score-field", args.score_field])
        if args.label_map:
            cmd.extend(["--label-map", args.label_map])
        cmd.extend(["--streaming", str(args.streaming)])

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    subprocess.run(cmd, check=True, cwd=project_dir)


if __name__ == "__main__":
    main()
