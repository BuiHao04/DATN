"""Select the top-N best-OCR images and build a fresh labeling set.

Ranking: keep images with >= --min-nodes detected nodes (i.e. a full receipt),
then sort by mean recognition score (descending) and take the top --top images.

Output is a self-contained labeling folder that mirrors the OCRLabelingPrepService
layout (nodes_to_label.csv with blank labels + images/ + ocr_json/), so it can be
opened directly in the step-3 labeling workspace.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

CSV_FIELDS = ["doc_id", "image_relpath", "text", "label", "x1", "y1", "x2", "y2", "score"]


def score_image(payload: dict) -> tuple[int, float]:
    nodes = payload.get("nodes", []) or []
    if not nodes:
        return 0, 0.0
    scores = [float(n.get("score", 0.0)) for n in nodes]
    return len(nodes), sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src/data/labeling_stage_b", help="Existing OCR output dir")
    ap.add_argument("--out", default="src/data/labeling_top1000", help="New labeling output dir")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--min-nodes", type=int, default=20)
    ap.add_argument("--copy-images", action="store_true", default=True)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    src_json = src / "ocr_json"
    src_images = src / "images"
    if not src_json.exists():
        raise SystemExit(f"No ocr_json dir: {src_json}")

    # Rank.
    ranked: list[tuple[float, int, Path, dict]] = []
    skipped_few = 0
    for jp in sorted(src_json.glob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip corrupt {jp.name}: {exc}")
            continue
        n_nodes, mean = score_image(payload)
        if n_nodes < args.min_nodes:
            skipped_few += 1
            continue
        ranked.append((mean, n_nodes, jp, payload))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = ranked[: args.top]
    print(
        f"candidates(>= {args.min_nodes} nodes)={len(ranked)} "
        f"skipped_few={skipped_few} selected={len(selected)}"
    )
    if selected:
        print(f"mean-score range: {selected[-1][0]:.4f} .. {selected[0][0]:.4f}")

    # Build output.
    out_json = out / "ocr_json"
    out_images = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    report: list[dict] = []
    for mean, n_nodes, jp, payload in selected:
        doc_id = str(payload.get("doc_id", jp.stem))
        relpath = str(payload.get("image_relpath", ""))
        # copy ocr json
        shutil.copy2(jp, out_json / jp.name)
        # copy image
        if args.copy_images and relpath:
            src_img = src_images / relpath
            if src_img.exists():
                dst_img = out_images / relpath
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
        # rows with blank label
        for node in payload.get("nodes", []):
            bbox = node.get("bbox", [0, 0, 0, 0])
            rows.append(
                {
                    "doc_id": doc_id,
                    "image_relpath": relpath,
                    "text": str(node.get("text", "")),
                    "label": "",
                    "x1": f"{float(bbox[0]):.2f}",
                    "y1": f"{float(bbox[1]):.2f}",
                    "x2": f"{float(bbox[2]):.2f}",
                    "y2": f"{float(bbox[3]):.2f}",
                    "score": f"{float(node.get('score', 0.0)):.4f}",
                }
            )
        report.append({"doc_id": doc_id, "num_nodes": n_nodes, "mean_score": round(mean, 4)})

    csv_path = out / "nodes_to_label.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    (out / "selection_report.json").write_text(
        json.dumps(
            {
                "src": str(src),
                "top": args.top,
                "min_nodes": args.min_nodes,
                "selected_images": len(selected),
                "total_nodes": len(rows),
                "images": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # carry over the OCR manifest for provenance
    src_manifest = src / "ocr_run_manifest.json"
    if src_manifest.exists():
        shutil.copy2(src_manifest, out / "ocr_run_manifest.json")

    print(f"wrote {len(rows)} blank-label rows for {len(selected)} images -> {csv_path}")


if __name__ == "__main__":
    main()
