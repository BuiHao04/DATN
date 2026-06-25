from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FIELDS = ["doc_id", "image_relpath", "text", "label", "x1", "y1", "x2", "y2", "score"]
VALID_LABELS = {
    "OTHER",
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
}


@dataclass
class Node:
    idx: int
    row: dict[str, str]
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float
    line_idx: int = -1

    @property
    def text(self) -> str:
        return self.row.get("text", "")


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("đ", "d")
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(text))


def has_any(text: str, words: tuple[str, ...]) -> bool:
    n = norm(text)
    c = compact(text)
    return any(w in n or compact(w) in c for w in words)


HEADER_KEYWORDS = (
    "ten hang",
    "mat hang",
    "san pham",
    "don gia",
    "gia ban",
    "thanh tien",
    "so tien",
    "so luong",
    " sl ",
    "dvt",
)

FOOTER_KEYWORDS = (
    "xin cam on",
    "hen gap",
    "tong dai",
    "gop y",
    "khieu nai",
    "website",
    "hotline",
    "ban sao hd",
    "hddt",
    "bachhoaxanh.com",
    "tax invoice",
)


def to_float(value: str | None) -> float:
    try:
        return float(str(value or "").strip())
    except Exception:
        return 0.0


def digits(text: str) -> str:
    return re.sub(r"\D", "", str(text or ""))


def has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in str(text or ""))


def looks_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"\s*[-+]?\d[\d.,:/-]*\s*", str(text or "")))


def looks_money(text: str) -> bool:
    raw = str(text or "").strip()
    ds = digits(raw)
    if not ds:
        return False
    if not looks_numeric(raw):
        return False
    return len(ds) >= 3 or "," in raw or "." in raw


def looks_qty(text: str) -> bool:
    raw = str(text or "").strip()
    if not re.fullmatch(r"\d{1,2}(?:[.,]\d)?", raw):
        return False
    try:
        return 0 < float(raw.replace(",", ".")) <= 99
    except Exception:
        return False


def clean_text(text: str) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    replacements = {
        "HÒA ĐƠN": "HÓA ĐƠN",
        "HOA ĐƠN": "HÓA ĐƠN",
        "HÓA DON": "HÓA ĐƠN",
        "BẢN HÀNG": "BÁN HÀNG",
        "Thành tiến": "Thành tiền",
        "Thành tien": "Thành tiền",
        "Tőng": "Tổng",
        "T6ng": "Tổng",
        "T8ng": "Tổng",
        "Tién": "Tiền",
        "trã": "trả",
        "cám an": "cảm ơn",
        "hen găp lai": "hẹn gặp lại",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"(Ngày:\s*\d{1,2}/\d{1,2}/\d{4}-\d{1,2})[.](\d{2})", r"\1:\2", out)
    return out


def build_lines(nodes: list[Node]) -> list[list[Node]]:
    lines: list[list[Node]] = []
    for node in sorted(nodes, key=lambda n: (n.cy, n.cx)):
        best = -1
        best_delta = 10**9
        for idx, line in enumerate(lines):
            ref_y = sum(n.cy for n in line) / len(line)
            ref_h = max(max(n.h for n in line), node.h, 1.0)
            delta = abs(node.cy - ref_y)
            if delta <= ref_h * 0.58 and delta < best_delta:
                best = idx
                best_delta = delta
        if best >= 0:
            node.line_idx = best
            lines[best].append(node)
            lines[best].sort(key=lambda n: n.cx)
        else:
            node.line_idx = len(lines)
            lines.append([node])
    return lines


def line_text(line: list[Node]) -> str:
    return " ".join(n.text for n in sorted(line, key=lambda n: n.cx))


def find_header_line(lines: list[list[Node]]) -> int | None:
    best = None
    best_score = 0
    for idx, line in enumerate(lines):
        text = line_text(line)
        score = 0
        for kw in HEADER_KEYWORDS:
            if kw in f" {norm(text)} " or compact(kw) in compact(text):
                score += 1
        if score > best_score:
            best = idx
            best_score = score
    return best if best_score >= 2 else None


def find_total_start(lines: list[list[Node]], start: int) -> int | None:
    for idx in range(max(start, 0), len(lines)):
        text = line_text(lines[idx])
        if has_any(
            text,
            (
                "tong so",
                "tong cong",
                "tong tien",
                "tong thanh toan",
                "phai thanh toan",
                "tam tinh",
                "subtotal",
                "vat",
                "thue",
                "giam gia",
                "discount",
                "tien khach",
                "tien tra",
                "khach tra",
            ),
        ):
            return idx
    return None


def first_alpha_line(lines: list[list[Node]]) -> int:
    for idx, line in enumerate(lines):
        if any(has_alpha(n.text) for n in line):
            return idx
    return 0


def amount_label(text: str, ctx: str) -> str | None:
    if not (looks_money(text) or any(ch.isdigit() for ch in text)):
        return None
    if has_any(ctx, FOOTER_KEYWORDS):
        return None
    if has_any(ctx, ("tong so",)):
        return None
    if has_any(ctx, ("tien khach", "khach dua", "khach tra", "tien thoi", "thoi lai", "tra lai")):
        return None
    if has_any(ctx, ("giam gia", "discount")):
        return "DISCOUNT"
    if has_any(ctx, ("phu thu", "service fee", "service charge")):
        return "SERVICE_FEE"
    if has_any(ctx, ("vat", "thue")):
        return "TAX_AMOUNT"
    if has_any(ctx, ("tam tinh", "subtotal")):
        return "SUBTOTAL"
    if has_any(ctx, ("tong thanh tien",)):
        return "SUBTOTAL"
    if has_any(ctx, ("tong cong", "tong tien", "phai thanh toan", "thanh toan", "total")):
        return "TOTAL_AMOUNT"
    return None


def scalar_label(node: Node, ctx: str, first_line: int, max_y: float) -> str | None:
    text = node.text
    y_norm = node.cy / max(max_y, 1.0)
    if has_any(text, FOOTER_KEYWORDS):
        return "OTHER"
    if has_any(text, ("phieu thanh toan bach hoa xanh", "bach hoa xanh")):
        return "MERCHANT_NAME"
    if has_any(text, ("hoa don", "bien lai", "phieu", "invoice")):
        return "OTHER"
    if has_any(text, HEADER_KEYWORDS):
        return "OTHER"
    if node.line_idx == first_line and has_alpha(text) and not any(ch.isdigit() for ch in text):
        return "MERCHANT_NAME"
    if node.line_idx <= first_line + 2 and has_alpha(text) and (
        has_any(ctx, ("cho ", "duong", "phuong", "quan", "huyen", "tp", "dia chi", "lam", "thu duc"))
        or (
            y_norm < 0.25
            and not any(ch.isdigit() for ch in text)
            and not has_any(text, ("hoa don", "bien lai", "phieu", "invoice", "ten hang"))
        )
    ):
        return "MERCHANT_ADDRESS"
    if has_any(ctx, ("sdt", "tel", "dien thoai", "hotline")) or re.search(r"\b0\d{8,10}\b", text):
        return "MERCHANT_PHONE"
    if has_any(ctx, ("ma so thue", "mst")):
        return "TAX_CODE" if any(ch.isdigit() for ch in text) or has_any(text, ("mst", "ma so thue")) else "OTHER"
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        return "DATE"
    if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text):
        return "TIME"
    if has_any(ctx, ("thu ngan", "cashier", "nhan vien")):
        return "CASHIER" if has_alpha(text) or any(ch.isdigit() for ch in text) else "OTHER"
    if has_any(ctx, ("so hd", "so gd", "invoice", "bill no", "receipt no", "so bien lai")):
        return "INVOICE_ID" if any(ch.isdigit() for ch in text) or has_any(text, ("so hd", "so gd", "invoice")) else "OTHER"
    if has_any(text, ("vnd", "tien mat", "cash", "momo", "visa", "mastercard", "zalopay", "chuyen khoan")):
        return "PAYMENT_METHOD"
    return None


def item_label(node: Node, line: list[Node], max_x: float) -> str:
    text = node.text
    ctx = line_text(line)
    x_norm = node.cx / max(max_x, 1.0)
    money_nodes = sorted([n for n in line if looks_money(n.text)], key=lambda n: n.cx)
    if has_any(text, FOOTER_KEYWORDS):
        return "OTHER"
    if has_any(ctx, HEADER_KEYWORDS) and len(line) <= 5:
        return "OTHER"
    if has_any(text, ("gia dang ky", "gia ban")):
        return "OTHER"
    if has_any(ctx, ("tong so", "tong cong", "tong tien", "tong thanh tien", "phai thanh toan", "tien khach", "tien thoi", "thoi lai", "tra lai")):
        return amount_label(text, ctx) or "OTHER"
    if has_alpha(text) and not looks_money(text):
        if has_any(text, ("hop", "cai", "chai", "goi", "lon", "ly", "kg", "suat")) and len(compact(text)) <= 8:
            return "OTHER"
        return "ITEM_NAME"
    if looks_qty(text) and x_norm < 0.6:
        return "ITEM_QTY"
    if looks_money(text):
        if has_any(ctx, ("gia dang ky", "gia ban", "don gia")):
            return "ITEM_UNIT_PRICE"
        if money_nodes and node is money_nodes[-1]:
            return "ITEM_AMOUNT"
        return "ITEM_UNIT_PRICE"
    if any(ch.isdigit() for ch in text) and (has_alpha(text) or "/" in text):
        return "ITEM_NAME"
    return "OTHER"


def label_doc(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str], int]:
    nodes: list[Node] = []
    for idx, row in enumerate(rows):
        row = dict(row)
        row["text"] = clean_text(row.get("text", ""))
        x1, y1, x2, y2 = (to_float(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        nodes.append(Node(idx, row, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1), max(y2 - y1, 1)))

    lines = build_lines(nodes)
    header = find_header_line(lines)
    total = find_total_start(lines, (header or 0) + 1)
    first_line = first_alpha_line(lines)
    max_x = max((n.x2 for n in nodes), default=1.0)
    max_y = max((n.y2 for n in nodes), default=1.0)
    line_ctx = {idx: line_text(line) for idx, line in enumerate(lines)}
    out = [dict(n.row) for n in nodes]
    counts: Counter[str] = Counter()
    changed = 0

    for node in nodes:
        text = node.text
        ctx = line_ctx.get(node.line_idx, text)
        old = str(out[node.idx].get("label", "")).strip()
        label = "OTHER"

        if has_any(ctx, FOOTER_KEYWORDS):
            label = "OTHER"
            scalar = None
        else:
            scalar = scalar_label(node, ctx, first_line, max_y)
        if scalar:
            label = scalar
        elif has_any(text, ("hoa don", "bien lai", "phieu", "invoice", "ten hang", "don gia", "gia ban", "thanh tien", "dvt")):
            label = "OTHER"
        elif total is not None and node.line_idx >= total:
            label = amount_label(text, ctx) or scalar_label(node, ctx, first_line, max_y) or "OTHER"
        elif (
            (header is not None and node.line_idx > header and (total is None or node.line_idx < total))
            or (header is None and total is not None and first_line <= node.line_idx < total)
        ):
            label = item_label(node, lines[node.line_idx], max_x)
        else:
            label = amount_label(text, ctx) or "OTHER"

        if label not in VALID_LABELS:
            label = "OTHER"
        out[node.idx]["text"] = clean_text(text)
        out[node.idx]["label"] = label
        counts[label] += 1
        changed += int(old != label)

    return out, counts, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    report_json = Path(args.report_json)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id not in grouped:
            grouped[doc_id] = []
            order.append(doc_id)
        grouped[doc_id].append(row)

    all_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    changed_total = 0
    doc_summaries = {}
    for doc_id in order:
        labeled, doc_counts, changed = label_doc(grouped[doc_id])
        all_rows.extend(labeled)
        counts.update(doc_counts)
        changed_total += changed
        if len(doc_summaries) < 30:
            doc_summaries[doc_id] = {"nodes": len(labeled), "labels": dict(doc_counts), "changed": changed}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in FIELDS} for row in all_rows])

    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "num_rows": len(all_rows),
        "num_docs": len(order),
        "changed_labels": changed_total,
        "label_counts": dict(counts),
        "doc_sample": doc_summaries,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
