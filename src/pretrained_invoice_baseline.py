from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def _resolve_path(project_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (project_dir / p)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env", override=False)

    image_path = _resolve_path(
        project_dir,
        os.getenv("PRETRAIN_IMAGE_PATH", "data/anh_test.jpg"),
    )
    ocr_debug_image = _resolve_path(
        project_dir,
        os.getenv("PRETRAIN_OCR_DEBUG_IMAGE", "outputs/ocr_boxes_pretrained.jpg"),
    )
    output_json = _resolve_path(
        project_dir,
        os.getenv("PRETRAIN_OUTPUT_JSON", "outputs/pretrained_invoice_result.json"),
    )

    command = [
        sys.executable,
        "pipeline_runner.py",
        "pretrained",
        "--project-dir",
        str(project_dir),
        "--image",
        str(image_path),
        "--lang",
        "en",
        "--ocr-debug-image",
        str(ocr_debug_image),
        "--output-json",
        str(output_json),
    ]

    subprocess.run(command, check=True, cwd=project_dir)


if __name__ == "__main__":
    main()
