from __future__ import annotations

from dataclasses import dataclass


LABEL_MAP = {
    0: "OTHER",
    1: "DATE",
    2: "TAX_CODE",
    3: "TOTAL_AMOUNT",
    4: "PRODUCT_NAME",
    5: "UNIT_PRICE",
}

LABEL_TO_ID = {value: key for key, value in LABEL_MAP.items()}


@dataclass
class OCRNode:
    text: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float
