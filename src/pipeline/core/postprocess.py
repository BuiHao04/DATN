from __future__ import annotations

import re
from typing import Any, Dict, List

from pipeline.core.schema import LABEL_MAP, OCRNode


MONEY_RE = re.compile(r"[-+]?\d[\d.,]*")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    replacements = {
        "PHÓA": "HÓA",
        "TIENMAT": "TIỀN MẶT",
        "Tổng tiên": "Tổng tiền",
        "TỔNG TIẾN": "TỔNG TIỀN",
        "thanh toáng": "thanh toán",
        "thạnh toáng": "thanh toán",
        "Điện thoai": "Điện thoại",
        "OSĐT": "SĐT",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _is_money(text: str) -> bool:
    raw = text.strip()
    digits = re.sub(r"\D", "", raw)
    return bool(MONEY_RE.fullmatch(raw)) and (len(digits) >= 3 or "," in raw or "." in raw)


def _money_value(text: str) -> int | None:
    matches = MONEY_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return int(digits)


def _looks_unit_price(text: str) -> bool:
    value = _money_value(text)
    return value is not None and (value >= 100 or "," in text or "." in text)


def _bad_item_text(text: str) -> bool:
    lower = text.lower()
    return any(
        k in lower
        for k in [
            "hóa đơn",
            "hoa don",
            "biên lai",
            "tên hàng",
            "mặt hàng",
            "đơn giá",
            "thành tiền",
            "tổng",
            "tiền khách",
            "tiền mặt",
            "vnd",
            "thu ngân",
            "cảm ơn",
            "xin c",
            "hẹn gặp",
            "website",
            "hotline",
            "tax invoice",
        ]
    )


def _first_text(rows: list[dict[str, Any]]) -> str | None:
    return rows[0]["text"] if rows else None


def _best_money(rows: list[dict[str, Any]]) -> str | None:
    money_rows = [r for r in rows if _money_value(r["text"]) is not None]
    if not money_rows:
        return _first_text(rows)
    # Receipt totals are usually near the bottom and right side. Use y first,
    # then numeric value as a tie-breaker.
    money_rows.sort(key=lambda r: (r["cy"], _money_value(r["text"]) or 0, r["cx"]), reverse=True)
    return money_rows[0]["text"]


def _field_rows(nodes: List[OCRNode], labels: List[int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    max_y = max((node.y2 for node in nodes), default=1.0)
    max_x = max((node.x2 for node in nodes), default=1.0)
    fields: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in [
            "MERCHANT_NAME",
            "MERCHANT_ADDRESS",
            "MERCHANT_PHONE",
            "DATE",
            "TIME",
            "TAX_CODE",
            "INVOICE_ID",
            "TOTAL_AMOUNT",
            "ITEM_NAME",
            "ITEM_QTY",
            "ITEM_UNIT_PRICE",
            "ITEM_AMOUNT",
            "SUBTOTAL",
            "SERVICE_FEE",
            "DISCOUNT",
            "TAX_AMOUNT",
            "PAYMENT_METHOD",
        ]
    }

    labeled_nodes: list[dict[str, Any]] = []
    for idx, (node, label_id) in enumerate(zip(nodes, labels)):
        label_name = LABEL_MAP.get(label_id, "OTHER")
        text = _clean_text(node.text)
        row = {
            "node_id": idx,
            "text": text,
            "label": label_name,
            "bbox": [node.x1, node.y1, node.x2, node.y2],
            "score": node.score,
            "cx": node.cx,
            "cy": node.cy,
            "x_norm": node.cx / max(max_x, 1.0),
            "y_norm": node.cy / max(max_y, 1.0),
        }
        labeled_nodes.append(row)
        if label_name in fields:
            fields[label_name].append(row)
    for rows in fields.values():
        rows.sort(key=lambda r: (r["cy"], r["cx"]))
    return fields, labeled_nodes


def _extract_item_names(rows: list[dict[str, Any]]) -> list[str]:
    cleaned = []
    seen = set()
    for row in rows:
        text = row["text"]
        if row["y_norm"] < 0.25 or row["y_norm"] > 0.82:
            continue
        if _bad_item_text(text):
            continue
        if _is_money(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def build_invoice_json(nodes: List[OCRNode], labels: List[int]) -> Dict[str, Any]:
    fields, labeled_nodes = _field_rows(nodes, labels)

    dates = [m.group(0) for row in fields["DATE"] for m in DATE_RE.finditer(row["text"])]
    times = [m.group(0) for row in fields["TIME"] + fields["DATE"] for m in TIME_RE.finditer(row["text"])]
    phone_rows = fields["MERCHANT_PHONE"]
    phone_values = []
    for row in phone_rows:
        matches = re.findall(r"0\d{8,10}", row["text"])
        phone_values.extend(matches or [row["text"]])

    item_names = _extract_item_names(fields["ITEM_NAME"])

    return {
        "invoice": {
            "merchant_name": _first_text(fields["MERCHANT_NAME"]),
            "merchant_address": _first_text(fields["MERCHANT_ADDRESS"]),
            "merchant_phone": phone_values[0] if phone_values else None,
            "date": dates[0] if dates else _first_text(fields["DATE"]),
            "time": times[0] if times else _first_text(fields["TIME"]),
            "tax_code": _first_text(fields["TAX_CODE"]),
            "invoice_id": _first_text(fields["INVOICE_ID"]),
            "subtotal": _best_money(fields["SUBTOTAL"]),
            "service_fee": _best_money(fields["SERVICE_FEE"]),
            "discount": _best_money(fields["DISCOUNT"]),
            "tax_amount": _best_money(fields["TAX_AMOUNT"]),
            "total_amount": _best_money(fields["TOTAL_AMOUNT"]),
            "payment_method": _first_text(fields["PAYMENT_METHOD"]),
            "item_name": item_names,
            "item_qty": [row["text"] for row in fields["ITEM_QTY"] if row["y_norm"] > 0.25],
            "item_unit_price": [row["text"] for row in fields["ITEM_UNIT_PRICE"] if _looks_unit_price(row["text"])],
            "item_amount": [row["text"] for row in fields["ITEM_AMOUNT"] if _money_value(row["text"]) is not None],
        },
        "nodes": [
            {
                "text": row["text"],
                "label": row["label"],
                "bbox": row["bbox"],
                "score": row["score"],
            }
            for row in labeled_nodes
        ],
    }
