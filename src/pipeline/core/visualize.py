from __future__ import annotations

from typing import List

import cv2
import numpy as np

from pipeline.core.schema import OCRNode


def draw_boxes_on_array(img, nodes: List[OCRNode], output_path: str) -> None:
    canvas = img.copy()
    for item in nodes:
        x1, y1, x2, y2 = int(item.x1), int(item.y1), int(item.x2), int(item.y2)
        text = item.text[:25]
        if item.quad and len(item.quad) >= 4:
            pts = np.array([[int(x), int(y)] for x, y in item.quad], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(output_path, canvas)


def draw_boxes(image_path: str, nodes: List[OCRNode], output_path: str) -> None:
    img = cv2.imread(image_path)
    draw_boxes_on_array(img, nodes, output_path)
