#!/usr/bin/env python3
"""A/B compare OCR recognizers (Paddle vs VietOCR hybrid) on sample images.

Run with the datn_hao env python and PYTHONPATH=src:
    PYTHONPATH=src /home/trung/miniconda3/envs/datn_hao/bin/python scripts/ab_ocr.py <img1> [img2 ...]
"""
from __future__ import annotations

import sys
import time

from pipeline.core.ocr_engine import run_ocr


def dump(title: str, nodes) -> None:
    print(f"\n===== {title}  ({len(nodes)} nodes) =====")
    for n in sorted(nodes, key=lambda x: (round(x.cy / 10), x.x1)):
        print(f"  [{n.score:.2f}] {n.text}")


def main() -> None:
    images = sys.argv[1:] or [
        "src/data/labeling_stage_b/images/batch_20260527_181547/00000004.jpg"
    ]
    for img in images:
        print("\n" + "#" * 70)
        print("IMAGE:", img)
        t0 = time.time()
        paddle_nodes = run_ocr(img, lang="vi", engine="paddle")
        t1 = time.time()
        dump(f"PADDLE  ({t1 - t0:.1f}s)", paddle_nodes)

        t0 = time.time()
        viet_nodes = run_ocr(img, lang="vi", engine="vietocr")
        t1 = time.time()
        dump(f"VIETOCR ({t1 - t0:.1f}s)", viet_nodes)


if __name__ == "__main__":
    main()
