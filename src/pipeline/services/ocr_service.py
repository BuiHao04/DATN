from __future__ import annotations

from pathlib import Path
from typing import List

from pipeline.core.ocr_engine import run_ocr
from pipeline.core.schema import OCRNode
from pipeline.core.visualize import draw_boxes


class OCRService:
    def run(self, image_path: str, lang: str = "vi", engine: str | None = None) -> List[OCRNode]:
        return run_ocr(image_path, lang=lang, engine=engine)

    def save_debug_image(self, image_path: str, nodes: List[OCRNode], output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        draw_boxes(image_path, nodes, output_path)
        return output_path
