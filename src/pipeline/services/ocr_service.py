from __future__ import annotations

from pathlib import Path
from typing import List

from pipeline.core.ocr_engine import prepare_ocr_image, run_ocr
from pipeline.core.schema import OCRNode
from pipeline.core.visualize import draw_boxes, draw_boxes_on_array


class OCRService:
    def run(
        self,
        image_path: str,
        lang: str = "vi",
        engine: str | None = None,
        overrides: dict | None = None,
    ) -> List[OCRNode]:
        return run_ocr(image_path, lang=lang, engine=engine, overrides=overrides)

    def save_debug_image(self, image_path: str, nodes: List[OCRNode], output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        draw_boxes(image_path, nodes, output_path)
        return output_path

    def prepare_image(self, image_path: str, overrides: dict | None = None):
        return prepare_ocr_image(image_path, overrides=overrides)

    def save_debug_image_from_array(self, image, nodes: List[OCRNode], output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        draw_boxes_on_array(image, nodes, output_path)
        return output_path
