from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "pipeline_runner.py",
            "pretrained",
            "--project-dir",
            ".",
            "--image",
            r"C:\Users\PC\Documents\datn_hao\invoice_ocr_gcn_demo\data\anh_test.jpg",
            "--lang",
            "en",
            "--ocr-debug-image",
            "outputs/ocr_boxes_pretrained.jpg",
            "--output-json",
            "outputs/pretrained_invoice_result.json",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
