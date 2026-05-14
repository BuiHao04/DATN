import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "outputs" / "ocr_result.json"
OUT_TXT = BASE_DIR / "outputs" / "invoice_final.txt"
OUT_MD = BASE_DIR / "outputs" / "invoice_final.md"

MONEY_RE = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\b")
DATE_RE = re.compile(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})")


def normalize_money_to_int(raw: str | None) -> int | None:
    if not raw:
        return None
    m = MONEY_RE.search(raw)
    if not m:
        return None
    # Invoice values are mostly thousand-grouped amounts, keep digits only.
    digits = re.sub(r"\D", "", m.group(0))
    return int(digits) if digits else None


def format_vnd(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}".replace(",", ".") + " đ"


def pick_best_date(invoice: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    candidates = []
    if invoice.get("date"):
        candidates.append(str(invoice["date"]))
    candidates.extend(str(n.get("text", "")) for n in nodes)

    for text in candidates:
        m = DATE_RE.search(text)
        if m:
            y, mm, dd = m.group(1), int(m.group(2)), int(m.group(3))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return f"{y}-{mm:02d}-{dd:02d}"
    return "N/A"


def pick_best_tax_code(invoice: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    if invoice.get("tax_code"):
        return str(invoice["tax_code"])
    for n in nodes:
        text = str(n.get("text", ""))
        low = text.lower()
        if any(k in low for k in ["mst", "msi", "tax"]):
            m = re.search(r"\d{10,14}", text)
            if m:
                return m.group(0)
    for n in nodes:
        m = re.search(r"\b\d{10,14}\b", str(n.get("text", "")))
        if m:
            return m.group(0)
    return "N/A"


def pick_best_total(invoice: dict[str, Any], nodes: list[dict[str, Any]]) -> int | None:
    direct = normalize_money_to_int(invoice.get("total_amount"))
    if direct is not None:
        return direct

    amounts: list[tuple[int, float]] = []
    for n in nodes:
        txt = str(n.get("text", ""))
        bbox = n.get("bbox", [0, 0, 0, 0])
        x_right = float(bbox[2]) if len(bbox) >= 3 else 0.0
        val = normalize_money_to_int(txt)
        if val is not None and val >= 1000:
            amounts.append((val, x_right))

    if not amounts:
        return None

    # Prefer right-most money values (usually amount column), then max value.
    amounts.sort(key=lambda it: (it[1], it[0]), reverse=True)
    top_right = amounts[:20]
    return max(v for v, _ in top_right)


def infer_products_and_prices(invoice: dict[str, Any], nodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    products = list(invoice.get("product_name") or [])
    prices = list(invoice.get("unit_price") or [])

    if not products:
        for n in nodes:
            text = str(n.get("text", "")).strip()
            if re.match(r"^\d{3}\s+", text) and any(c.isalpha() for c in text):
                products.append(text)

    if not prices:
        vals = []
        for n in nodes:
            text = str(n.get("text", ""))
            for m in MONEY_RE.finditer(text):
                v = normalize_money_to_int(m.group(0))
                if v is not None and 1000 <= v <= 5_000_000:
                    vals.append(v)
        vals = sorted(set(vals))
        prices = [format_vnd(v) for v in vals[:30]]

    return products, prices


def build_lines(data: dict[str, Any]) -> tuple[str, str]:
    invoice = data.get("invoice", {})
    nodes = data.get("nodes", [])

    date = pick_best_date(invoice, nodes)
    tax_code = pick_best_tax_code(invoice, nodes)
    total_int = pick_best_total(invoice, nodes)
    products, prices = infer_products_and_prices(invoice, nodes)

    lines = [
        "HOA DON (OCR + GRAPH + GCN PIPELINE)",
        "=" * 40,
        f"Ngay hoa don: {date}",
        f"Ma so thue: {tax_code}",
        f"Tong cong: {format_vnd(total_int)}",
        "",
        "Danh sach mat hang:",
    ]

    if not products:
        lines.append("- N/A")
    else:
        for i, p in enumerate(products, start=1):
            lines.append(f"{i}. {p}")

    lines.extend(["", "Don gia tham khao:"])
    if not prices:
        lines.append("- N/A")
    else:
        for i, p in enumerate(prices, start=1):
            lines.append(f"{i}. {p}")

    txt = "\n".join(lines)

    md_lines = [
        "# HOA DON (OCR + GRAPH + GCN PIPELINE)",
        "",
        f"- **Ngay hoa don:** {date}",
        f"- **Ma so thue:** {tax_code}",
        f"- **Tong cong:** {format_vnd(total_int)}",
        "",
        "## Danh sach mat hang",
    ]

    if not products:
        md_lines.append("- N/A")
    else:
        for p in products:
            md_lines.append(f"- {p}")

    md_lines.append("")
    md_lines.append("## Don gia tham khao")
    if not prices:
        md_lines.append("- N/A")
    else:
        for p in prices:
            md_lines.append(f"- {p}")

    md = "\n".join(md_lines)
    return txt, md


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Khong tim thay file: {INPUT_JSON}")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    txt, md = build_lines(data)

    OUT_TXT.write_text(txt, encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")

    print("Da render xong:")
    print(f"- {OUT_TXT}")
    print(f"- {OUT_MD}")


if __name__ == "__main__":
    main()
