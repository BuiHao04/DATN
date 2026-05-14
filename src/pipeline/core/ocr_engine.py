from __future__ import annotations

from typing import List

from paddleocr import PaddleOCR

from pipeline.core.schema import OCRNode


def run_ocr(image_path: str, lang: str = "en") -> List[OCRNode]:
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    # PaddleOCR API differs by version:
    # - old: ocr.ocr(image_path, cls=True)
    # - new: ocr.ocr(image_path)
    try:
        result = ocr.ocr(image_path, cls=True)
    except TypeError:
        result = ocr.ocr(image_path)

    nodes: List[OCRNode] = []
    if not result:
        return nodes

    lines = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result

    for line in lines:
        if not line or len(line) < 2:
            continue

        box = line[0]
        text = line[1][0]
        score = float(line[1][1])

        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x1, y1 = float(min(x_coords)), float(min(y_coords))
        x2, y2 = float(max(x_coords)), float(max(y_coords))

        nodes.append(
            OCRNode(
                text=text,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
            )
        )
    return nodes
