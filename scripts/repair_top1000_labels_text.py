from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path("src/data/labeling_top1000")
CSV_PATH = BASE_DIR / "nodes_to_label.csv"
BACKUP_PATH = BASE_DIR / "nodes_to_label.before_repair.csv"
REPORT_PATH = BASE_DIR / "nodes_to_label.repair_report.json"
FIELDS = ["doc_id", "image_relpath", "text", "label", "x1", "y1", "x2", "y2", "score"]
VALID_LABELS = {
    "MERCHANT_NAME",
    "MERCHANT_ADDRESS",
    "MERCHANT_PHONE",
    "TAX_CODE",
    "INVOICE_ID",
    "DATE",
    "TIME",
    "CASHIER",
    "ITEM_NAME",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_AMOUNT",
    "SUBTOTAL",
    "SERVICE_FEE",
    "DISCOUNT",
    "TAX_AMOUNT",
    "TOTAL_AMOUNT",
    "PAYMENT_METHOD",
    "OTHER",
}
LABEL_ALIASES = {
    "MERC": "MERCHANT_NAME",
    "PRODUCT_NAME": "ITEM_NAME",
    "UNIT_PRICE": "ITEM_UNIT_PRICE",
}


TEXT_REPLACEMENTS = {
    "OSĐT": "SĐT",
    "Điện thoai": "Điện thoại",
    "TIENMAT": "TIỀN MẶT",
    "TỔNG TIẾN": "TỔNG TIỀN",
    "Tổng tiên": "Tổng tiền",
    "Tổng Tiên": "Tổng Tiền",
    "GIÁM GIÁO": "GIẢM GIÁ",
    "giám giáo": "giảm giá",
    "giảm giáo": "giảm giá",
    "giảm gia": "giảm giá",
    "PHUTHU": "PHỤ THU",
    "TIÊN TRẢ LẠI": "TIỀN TRẢ LẠI",
    "Chi tiết thanh toáng": "Chi tiết thanh toán",
    "thanh toáng": "thanh toán",
    "thạnh toáng": "thanh toán",
    "Thời giang": "Thời gian",
    "Passwhi": "Pass wifi",
    "Cảm on": "Cảm ơn",
    "naychỉ": "nay chỉ",
    "Điện thoai:": "Điện thoại:",
    "TIỀN MẶTO": "TIỀN MẶT",
}


def clean_text(text: str, label: str = "") -> str:
    out = str(text or "").strip()
    for src, dst in TEXT_REPLACEMENTS.items():
        out = out.replace(src, dst)
    if label == "MERCHANT_NAME":
        out = out.replace(" Food 8 Drink", " Food & Drink")
        out = out.replace("Food 8 Drink", "Food & Drink")
    return re.sub(r"\s+", " ", out).strip()


def normalize_label(label: str) -> str | None:
    value = str(label or "").strip().upper()
    if value in LABEL_ALIASES:
        return LABEL_ALIASES[value]
    if value in VALID_LABELS:
        return value
    return None


def to_float(raw: str | None) -> float | None:
    try:
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)
    except ValueError:
        return None


def bbox_key(doc_id: str, bbox: list[Any]) -> tuple[str, str, str, str, str] | None:
    if len(bbox) < 4:
        return None
    nums = [to_float(str(v)) for v in bbox[:4]]
    if any(v is None for v in nums):
        return None
    return (doc_id, *(f"{v:.2f}" for v in nums if v is not None))


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str] | None:
    nums = [to_float(row.get(k)) for k in ("x1", "y1", "x2", "y2")]
    if any(v is None for v in nums):
        return None
    return (row.get("doc_id", ""), *(f"{v:.2f}" for v in nums if v is not None))


def infer_label(doc_nodes: list[dict[str, Any]], idx: int, text: str) -> str:
    lower = text.lower()
    compact = re.sub(r"\s+", "", lower)
    prev_text = " ".join(str(n.get("text", "")) for n in doc_nodes[max(0, idx - 3):idx]).lower()
    next_text = " ".join(str(n.get("text", "")) for n in doc_nodes[idx + 1:idx + 4]).lower()
    context = f"{prev_text} {lower} {next_text}"
    looks_amount = bool(re.search(r"^\s*[-+]?\d[\d.,]*\s*$", text))
    if looks_amount:
        if "vat" in context or "thuế" in context:
            return "TAX_AMOUNT"
        if "giảm giá" in context or "discount" in context:
            return "DISCOUNT"
        if "phụ thu" in context or "service fee" in context:
            return "SERVICE_FEE"
        if "tạm tính" in context or "subtotal" in context:
            return "SUBTOTAL"
        if any(k in context for k in ("tổng tiền", "tổng cộng", "thành tiền", "total")):
            return "TOTAL_AMOUNT"
    if idx == 0 and not any(k in lower for k in ("hóa đơn", "hoa don", "biên lai", "phieu", "phiếu")):
        return "MERCHANT_NAME"
    if "mã số thuế" in lower or lower.startswith("mst") or " mst" in lower:
        return "TAX_CODE"
    if "sđt" in lower or "điện thoại" in lower or re.search(r"(?:^|\\D)(0\\d{8,10})(?:\\D|$)", text):
        return "MERCHANT_PHONE"
    if re.search(r"\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b", text):
        return "DATE"
    if re.search(r"\\b\\d{1,2}:\\d{2}(?::\\d{2})?\\b", text):
        return "TIME"
    if "thu ngân" in lower or "cashier" in lower:
        return "CASHIER"
    if any(k in lower for k in ("tiền mặt", "tien mat", "cash", "momo", "visa", "mastercard", "zalopay", "chuyển khoản")):
        return "PAYMENT_METHOD"
    if "giảm giá" in lower or "discount" in lower:
        return "DISCOUNT"
    if "phụ thu" in lower or "service fee" in lower:
        return "SERVICE_FEE"
    if "vat" in lower or "thuế" in lower:
        return "TAX_AMOUNT"
    if any(k in lower for k in ("thành tiền", "tổng tiền", "tổng cộng", "total amount", "grand total")):
        return "TOTAL_AMOUNT"
    if "tạm tính" in lower or "subtotal" in lower:
        return "SUBTOTAL"
    if any(k in compact for k in ("sốhđ", "sohd", "sốhoáđơn", "sốhóađơn", "sốbiênlai")):
        return "INVOICE_ID"
    return "OTHER"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)
    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        old_rows = list(csv.DictReader(f))

    labels_by_bbox: dict[tuple[str, str, str, str, str], str] = {}
    text_by_bbox: dict[tuple[str, str, str, str, str], str] = {}
    invalid_label_rows: list[dict[str, Any]] = []
    malformed_rows = 0
    for row_number, row in enumerate(old_rows, start=2):
        key = row_key(row)
        label = normalize_label(row.get("label", ""))
        if key is None:
            malformed_rows += 1
            continue
        if label is None:
            invalid_label_rows.append(
                {
                    "row_number": row_number,
                    "doc_id": row.get("doc_id", ""),
                    "text": row.get("text", ""),
                    "label": row.get("label", ""),
                }
            )
            continue
        labels_by_bbox[key] = label
        text_by_bbox[key] = row.get("text", "")

    repaired_rows: list[dict[str, str]] = []
    changed_text: list[dict[str, Any]] = []
    changed_label: list[dict[str, Any]] = []
    unmatched_nodes: list[dict[str, Any]] = []

    for json_path in sorted((BASE_DIR / "ocr_json").glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        doc_id = str(payload.get("doc_id") or json_path.stem)
        image_relpath = str(payload.get("image_relpath") or "")
        nodes = payload.get("nodes") or []
        for idx, node in enumerate(nodes):
            bbox = node.get("bbox") or [0, 0, 0, 0]
            key = bbox_key(doc_id, bbox)
            raw_text = str(node.get("text", ""))
            label = labels_by_bbox.get(key) if key else None
            if label is None:
                label = infer_label(nodes, idx, raw_text)
                unmatched_nodes.append({"doc_id": doc_id, "node_index": idx, "text": raw_text, "label": label})
            old_text = text_by_bbox.get(key) if key else None
            text_source = old_text if old_text is not None else raw_text
            text = clean_text(text_source, label)
            if old_text is not None and str(old_text).strip() != text:
                changed_text.append({"doc_id": doc_id, "node_index": idx, "old": old_text, "new": text})
            old_label = labels_by_bbox.get(key) if key else None
            if old_label is not None and old_label != label:
                changed_label.append({"doc_id": doc_id, "node_index": idx, "old": old_label, "new": label, "text": text})
            repaired_rows.append(
                {
                    "doc_id": doc_id,
                    "image_relpath": image_relpath,
                    "text": text,
                    "label": label,
                    "x1": f"{float(bbox[0]):.2f}",
                    "y1": f"{float(bbox[1]):.2f}",
                    "x2": f"{float(bbox[2]):.2f}",
                    "y2": f"{float(bbox[3]):.2f}",
                    "score": f"{float(node.get('score', 0.0)):.4f}",
                }
            )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(repaired_rows)

    report = {
        "input_csv": str(CSV_PATH),
        "backup_csv": str(BACKUP_PATH),
        "output_rows": len(repaired_rows),
        "output_docs": len({r["doc_id"] for r in repaired_rows}),
        "old_rows": len(old_rows),
        "old_malformed_rows": malformed_rows,
        "old_invalid_label_rows": invalid_label_rows,
        "unmatched_nodes_relabelled_by_rules": len(unmatched_nodes),
        "unmatched_nodes_sample": unmatched_nodes[:50],
        "changed_text_count": len(changed_text),
        "changed_text_sample": changed_text[:50],
        "changed_label_count": len(changed_label),
        "changed_label_sample": changed_label[:50],
        "label_counts": dict(Counter(r["label"] for r in repaired_rows)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
