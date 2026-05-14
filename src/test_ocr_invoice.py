from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "pipeline_runner.py",
            "gcn_infer",
            "--image",
            r"C:\Users\PC\Documents\datn_hao\data_test\hoa-don-sieu-thi.jpg",
            "--lang",
            "en",
            "--ocr-debug-image",
            "outputs/ocr_boxes.jpg",
            "--output-json",
            "outputs/ocr_result.json",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
