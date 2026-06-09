from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


LABEL_MAP = {
    0: "OTHER",
    1: "MERCHANT_NAME",
    2: "MERCHANT_ADDRESS",
    3: "MERCHANT_PHONE",
    4: "TAX_CODE",
    5: "INVOICE_ID",
    6: "DATE",
    7: "TIME",
    8: "CASHIER",
    9: "ITEM_NAME",
    10: "ITEM_QTY",
    11: "ITEM_UNIT_PRICE",
    12: "ITEM_AMOUNT",
    13: "SUBTOTAL",
    14: "SERVICE_FEE",
    15: "DISCOUNT",
    16: "TAX_AMOUNT",
    17: "TOTAL_AMOUNT",
    18: "PAYMENT_METHOD",
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
    quad: Tuple[Tuple[float, float], ...] | None = None
