from __future__ import annotations

from typing import List

import cv2

from pipeline.core.schema import OCRNode


def draw_boxes(image_path: str, nodes: List[OCRNode], output_path: str) -> None:
    img = cv2.imread(image_path)
    for item in nodes:
        x1, y1, x2, y2 = map(int, [item.x1, item.y1, item.x2, item.y2])
        text = item.text[:25]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(output_path, img)
