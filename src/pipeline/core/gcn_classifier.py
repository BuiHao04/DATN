from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from pipeline.core.schema import LABEL_TO_ID, OCRNode

try:
    from torch_geometric.nn import GCNConv
except Exception:
    GCNConv = None


def build_features(nodes: List[OCRNode]) -> torch.Tensor:
    max_x = max((n.x2 for n in nodes), default=1.0)
    max_y = max((n.y2 for n in nodes), default=1.0)
    max_x = max(max_x, 1.0)
    max_y = max(max_y, 1.0)

    def line_groups() -> list[list[int]]:
        groups: list[list[int]] = []
        for node_idx in sorted(range(len(nodes)), key=lambda i: (nodes[i].cy, nodes[i].cx)):
            node = nodes[node_idx]
            best = -1
            best_delta = float("inf")
            for group_idx, group in enumerate(groups):
                ref_y = sum(nodes[i].cy for i in group) / max(len(group), 1)
                ref_h = max(max(nodes[i].h for i in group), node.h, 1.0)
                delta = abs(node.cy - ref_y)
                if delta <= ref_h * 0.58 and delta < best_delta:
                    best = group_idx
                    best_delta = delta
            if best >= 0:
                groups[best].append(node_idx)
                groups[best].sort(key=lambda i: nodes[i].cx)
            else:
                groups.append([node_idx])
        return groups

    lines = line_groups()
    line_by_node: dict[int, int] = {}
    position_by_node: dict[int, int] = {}
    for line_idx, line in enumerate(lines):
        for pos, node_idx in enumerate(line):
            line_by_node[node_idx] = line_idx
            position_by_node[node_idx] = pos

    def line_text(line: list[int]) -> str:
        return " ".join(nodes[i].text.lower() for i in line)

    line_texts = [line_text(line) for line in lines]

    def normalize_value(value: str) -> str:
        value = str(value or "").lower().replace("đ", "d")
        return "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )

    def compact_value(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", normalize_value(value))

    def has_kw(value: str, keywords: list[str]) -> bool:
        value = normalize_value(value)
        c = compact_value(value)
        return any(k in value or compact_value(k) in c for k in keywords)

    header_line = -1
    best_header_score = 0
    for idx, text in enumerate(line_texts):
        score = sum(
            1
            for kw in ["tên hàng", "ten hang", "mặt hàng", "mat hang", "đơn giá", "don gia", "thành tiền", "thanh tien", "dvt", "sl"]
            if has_kw(text, [kw])
        )
        if score > best_header_score:
            header_line = idx
            best_header_score = score
    if best_header_score < 2:
        header_line = -1

    total_line = -1
    for idx, text in enumerate(line_texts):
        if header_line >= 0 and idx <= header_line:
            continue
        if has_kw(text, ["tổng số", "tong so", "tổng cộng", "tong cong", "tổng tiền", "tong tien", "phải thanh toán", "phai thanh toan", "tạm tính", "tam tinh", "subtotal", "vat", "thuế", "thue", "giảm giá", "giam gia", "tiền khách", "tien khach"]):
            total_line = idx
            break

    features = []
    for idx, n in enumerate(nodes):
        text = n.text.lower()
        compact = re.sub(r"\s+", "", text)
        digits = re.sub(r"\D", "", text)
        y_norm = n.cy / max_y
        x_norm = n.cx / max_x
        digit_count = len(digits)
        numeric_value = float(digits[:9]) if digits else 0.0
        looks_money = 1.0 if re.fullmatch(r"\s*[-+]?\d[\d.,]*\s*", text) and (len(digits) >= 3 or "," in text or "." in text) else 0.0
        has_total_kw = 1.0 if any(k in text for k in ["tổng", "total", "phải t.toán", "phải thanh toán", "thanh toán"]) else 0.0
        has_item_header_kw = 1.0 if any(k in text for k in ["tên hàng", "mặt hàng", "đơn giá", "thành tiền", "đvt", "sl"]) else 0.0
        has_date = 1.0 if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text) else 0.0
        has_time = 1.0 if re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", text) else 0.0
        has_phone = 1.0 if re.search(r"\b0\d{8,10}\b", text) or any(k in text for k in ["sđt", "sdt", "tel", "điện thoại"]) else 0.0
        has_tax_kw = 1.0 if any(k in text for k in ["mst", "mã số thuế"]) else 0.0
        has_invoice_kw = 1.0 if any(k in text for k in ["số hđ", "số gd", "số biên lai", "invoice"]) else 0.0
        has_payment_kw = 1.0 if any(k in text for k in ["tiền mặt", "tien mat", "vnd", "momo", "visa", "mastercard"]) else 0.0
        has_subtotal_kw = 1.0 if any(k in text for k in ["tạm tính", "tam tinh", "subtotal"]) else 0.0
        has_discount_kw = 1.0 if any(k in text for k in ["giảm", "giam", "discount"]) else 0.0
        has_service_kw = 1.0 if any(k in text for k in ["phụ thu", "phu thu", "service fee"]) else 0.0
        has_cashier_kw = 1.0 if any(k in text for k in ["thu ngân", "thu ngan", "cashier", "nhân viên", "nhan vien"]) else 0.0
        has_address_kw = 1.0 if any(k in text for k in ["địa chỉ", "dia chi", "phường", "phuong", "quận", "quan", "huyện", "huyen", "tp.", "chợ ", "cho "]) else 0.0
        has_receipt_title_kw = 1.0 if any(k in text for k in ["hóa đơn", "hoa don", "biên lai", "bien lai", "phiếu", "phieu"]) else 0.0
        has_footer_kw = 1.0 if any(k in text for k in ["cảm ơn", "cam on", "hẹn gặp", "hen gap", "website", "hotline", "tax invoice"]) else 0.0
        has_unit_word = 1.0 if any(k in compact for k in ["hop", "chai", "goi", "cai", "lon", "ly", "kg", "suat"]) else 0.0
        qty_candidate = 1.0 if re.fullmatch(r"\d{1,2}([,.]\d)?", text.strip()) else 0.0
        long_numeric = 1.0 if digit_count >= 8 else 0.0
        is_top = 1.0 if y_norm < 0.22 else 0.0
        is_middle = 1.0 if 0.22 <= y_norm <= 0.78 else 0.0
        is_bottom = 1.0 if y_norm > 0.78 else 0.0
        is_left = 1.0 if x_norm < 0.35 else 0.0
        is_right = 1.0 if x_norm > 0.65 else 0.0
        is_center = 1.0 if 0.35 <= x_norm <= 0.65 else 0.0
        line_idx = line_by_node.get(idx, 0)
        pos_in_line = position_by_node.get(idx, 0)
        line = lines[line_idx] if 0 <= line_idx < len(lines) else [idx]
        line_len = max(len(line), 1)
        line_text_value = line_texts[line_idx] if 0 <= line_idx < len(line_texts) else text
        money_nodes_in_line = [i for i in line if re.fullmatch(r"\s*[-+]?\d[\d.,:/-]*\s*", nodes[i].text.lower()) and re.sub(r"\D", "", nodes[i].text)]
        rightmost_money_idx = max(money_nodes_in_line, key=lambda i: nodes[i].cx) if money_nodes_in_line else -1
        line_has_total_kw = 1.0 if has_kw(line_text_value, ["tổng số", "tong so", "tổng cộng", "tong cong", "tổng tiền", "tong tien", "phải thanh toán", "phai thanh toan", "total"]) else 0.0
        line_has_tax_kw = 1.0 if has_kw(line_text_value, ["vat", "thuế", "thue", "tax"]) else 0.0
        line_has_subtotal_kw = 1.0 if has_kw(line_text_value, ["tạm tính", "tam tinh", "subtotal"]) else 0.0
        line_has_payment_kw = 1.0 if has_kw(line_text_value, ["vnd", "tiền mặt", "tien mat", "cash", "momo", "visa", "chuyển khoản", "chuyen khoan"]) else 0.0
        line_has_invoice_kw = 1.0 if has_kw(line_text_value, ["số hđ", "so hd", "số gd", "so gd", "invoice", "bill no", "receipt no"]) else 0.0
        line_has_cashier_kw = 1.0 if has_kw(line_text_value, ["thu ngân", "thu ngan", "cashier", "nhân viên", "nhan vien"]) else 0.0
        line_has_phone_kw = 1.0 if has_kw(line_text_value, ["sđt", "sdt", "tel", "điện thoại", "dien thoai", "hotline"]) else 0.0
        line_has_date = 1.0 if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line_text_value) else 0.0
        line_has_time = 1.0 if re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", line_text_value) else 0.0
        in_item_region = 1.0 if header_line >= 0 and line_idx > header_line and (total_line < 0 or line_idx < total_line) else 0.0
        in_total_region = 1.0 if total_line >= 0 and line_idx >= total_line else 0.0
        is_rightmost_money = 1.0 if idx == rightmost_money_idx else 0.0
        features.append(
            [
                min(len(n.text), 64) / 64.0,
                1.0 if any(c.isdigit() for c in text) else 0.0,
                1.0 if any(k in text for k in [".", ",", "đ", "vnd"]) else 0.0,
                x_norm,
                y_norm,
                n.w / max_x,
                n.h / max_y,
                n.score,
                looks_money,
                has_total_kw,
                has_item_header_kw,
                has_date,
                has_time,
                has_phone,
                has_tax_kw,
                has_invoice_kw,
                has_payment_kw,
                is_top,
                is_middle,
                is_bottom,
                is_left,
                is_center,
                is_right,
                min(idx, 64) / 64.0,
                1.0 if any(c.isalpha() for c in text) else 0.0,
                1.0 if "vnd" in compact else 0.0,
                has_subtotal_kw,
                has_discount_kw,
                has_service_kw,
                has_cashier_kw,
                has_address_kw,
                has_receipt_title_kw,
                has_footer_kw,
                has_unit_word,
                qty_candidate,
                long_numeric,
                min(digit_count, 12) / 12.0,
                min(numeric_value, 1_000_000.0) / 1_000_000.0,
                1.0 if x_norm < 0.2 else 0.0,
                1.0 if 0.2 <= x_norm < 0.4 else 0.0,
                1.0 if 0.4 <= x_norm < 0.6 else 0.0,
                1.0 if 0.6 <= x_norm < 0.8 else 0.0,
                1.0 if x_norm >= 0.8 else 0.0,
                min(line_idx, 64) / 64.0,
                pos_in_line / max(line_len - 1, 1),
                min(line_len, 12) / 12.0,
                1.0 if pos_in_line == 0 else 0.0,
                1.0 if pos_in_line == line_len - 1 else 0.0,
                min(len(money_nodes_in_line), 5) / 5.0,
                is_rightmost_money,
                1.0 if line_idx == header_line else 0.0,
                in_item_region,
                in_total_region,
                line_has_total_kw,
                line_has_tax_kw,
                line_has_subtotal_kw,
                line_has_payment_kw,
                line_has_invoice_kw,
                line_has_cashier_kw,
                line_has_phone_kw,
                line_has_date,
                line_has_time,
            ]
        )
    return torch.tensor(features, dtype=torch.float32)


def build_edge_index(edges: List[Tuple[int, int, float]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    raw = [[src, dst] for src, dst, _ in edges]
    return torch.tensor(raw, dtype=torch.long).t().contiguous()


class InvoiceGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch-geometric is not installed")
        if num_layers < 2:
            raise ValueError("InvoiceGCN requires at least 2 layers")
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.hidden_convs = nn.ModuleList(
            GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 2)
        )
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.hidden_convs:
            residual = x
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape == residual.shape:
                x = x + residual
        x = self.conv2(x, edge_index)
        return x


def infer_gcn_hparams_from_state(
    state: dict[str, torch.Tensor],
    fallback_in_channels: int,
    fallback_out_channels: int,
) -> tuple[int, int, int, int]:
    conv1_weight = state.get("conv1.lin.weight")
    conv1_bias = state.get("conv1.bias")
    conv2_bias = state.get("conv2.bias")

    in_channels = int(conv1_weight.shape[1]) if conv1_weight is not None and conv1_weight.ndim == 2 else fallback_in_channels
    hidden_channels = int(conv1_bias.shape[0]) if conv1_bias is not None and conv1_bias.ndim == 1 else 64
    out_channels = int(conv2_bias.shape[0]) if conv2_bias is not None and conv2_bias.ndim == 1 else fallback_out_channels
    hidden_layer_indices = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"hidden_convs\.(\d+)\.", key))
    }
    num_layers = 2 + (max(hidden_layer_indices) + 1 if hidden_layer_indices else 0)
    return in_channels, hidden_channels, out_channels, num_layers


def classify_nodes(nodes: List[OCRNode], edges: List[Tuple[int, int, float]]) -> List[int]:
    if not nodes:
        return []

    neighbor_map: dict[int, List[int]] = {i: [] for i in range(len(nodes))}
    for src, dst, _ in edges:
        neighbor_map[src].append(dst)

    max_y = max((n.y2 for n in nodes), default=1.0)
    sorted_by_y = sorted(range(len(nodes)), key=lambda idx: (nodes[idx].cy, nodes[idx].cx))
    top_indices = set(sorted_by_y[:3])

    def looks_amount(value: str) -> bool:
        raw = value.strip()
        return bool(re.fullmatch(r"[-+]?\d[\d.,]*", raw)) and any(c.isdigit() for c in raw)

    def looks_money(value: str) -> bool:
        raw = value.strip()
        digits = re.sub(r"\D", "", raw)
        return looks_amount(raw) and (len(digits) >= 4 or "," in raw or "." in raw)

    def normalize_value(value: str) -> str:
        value = str(value or "").lower().replace("đ", "d")
        return "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )

    def compact_value(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", normalize_value(value))

    def has_keyword(value: str, keywords: list[str]) -> bool:
        value = normalize_value(value)
        compact_value_text = compact_value(value)
        return any(k in value or compact_value(k) in compact_value_text for k in keywords)

    labels: List[int] = []
    for i, node in enumerate(nodes):
        raw_text = node.text.strip()
        text = normalize_value(raw_text)
        compact = re.sub(r"\s+", "", text)
        neighbor_text = " ".join(normalize_value(nodes[j].text) for j in neighbor_map[i][:8])
        context = f"{text} {neighbor_text}"
        y_norm = node.cy / max(max_y, 1.0)

        if not text:
            label = "OTHER"
        elif has_keyword(text, ["hoa don", "bien lai", "phieu", "phieu tinh tien"]):
            label = "OTHER"
        elif has_keyword(text, ["ten hang", "mat hang", "don gia", "gia ban", "thanh tien", "so tien", "dvt", "sl"]):
            label = "OTHER"
        elif has_keyword(text, ["xin c", "cam on", "hen gap", "website", "hotline", "tong dai", "gop y", "khieu nai", "hddt", "tax invoice"]):
            label = "OTHER"
        elif i == sorted_by_y[0] and any(c.isalpha() for c in text) and not any(c.isdigit() for c in text):
            label = "MERCHANT_NAME"
        elif (len(sorted_by_y) > 1 and i == sorted_by_y[1] and any(c.isalpha() for c in text) and not any(c.isdigit() for c in text)) or has_keyword(text, ["địa chỉ", "dia chi", "phường", "quận", "huyện", "tp.", "thành phố", "khu ", "chợ "]):
            label = "MERCHANT_ADDRESS"
        elif has_keyword(text, ["sdt", "tel", "dien thoai"]) or re.search(r"\b0\d{8,10}\b", text):
            label = "MERCHANT_PHONE"
        elif has_keyword(text, ["ma so thue", "mst"]) or (
            text.replace("/", "").isdigit() and len(text) in {8, 10, 12, 13, 14} and "mst" in context
        ):
            label = "TAX_CODE"
        elif has_keyword(text, ["thu ngan", "cashier"]):
            label = "CASHIER"
        elif any(c.isdigit() for c in text) and ("ngay" in text or re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)):
            label = "DATE"
        elif re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", text):
            label = "TIME"
        elif has_keyword(text, ["số hđ", "so hd", "số gd", "số biên lai", "invoice"]):
            label = "INVOICE_ID"
        elif has_keyword(text, ["tien mat", "vnd", "momo", "visa", "mastercard", "chuyen khoan"]):
            label = "PAYMENT_METHOD"
        elif any(c.isdigit() for c in text) and has_keyword(context, ["giam gia", "discount"]):
            label = "DISCOUNT"
        elif any(c.isdigit() for c in text) and has_keyword(context, ["phu thu", "service fee"]):
            label = "SERVICE_FEE"
        elif any(c.isdigit() for c in text) and has_keyword(context, ["vat", "thue"]):
            label = "TAX_AMOUNT"
        elif looks_money(text) and has_keyword(context, ["tien khach", "khach dua", "khach tra", "tien thoi", "thoi lai", "tra lai"]):
            label = "OTHER"
        elif looks_money(text) and has_keyword(context, ["tong cong", "tong tien", "phai thanh toan", "phai t.toan", "total"]):
            label = "TOTAL_AMOUNT"
        elif has_keyword(text, ["tong cong", "tong so", "tong tien", "tong thanh tien", "tien khach", "tien tra", "tien thoi", "phai thanh toan", "phai t.toan"]):
            label = "OTHER"
        elif any(c.isdigit() for c in text) and any(c.isalpha() for c in text) and 0.25 <= y_norm <= 0.75:
            label = "ITEM_NAME"
        elif text.isdigit() and 1 <= int(text) <= 99 and any(k in context for k in ["sl", "s.lượng", "số lượng"]):
            label = "ITEM_QTY"
        elif any(c.isdigit() for c in text) and has_keyword(context, ["đơn giá", "unit", "price"]):
            label = "ITEM_UNIT_PRICE"
        elif looks_money(text) and y_norm > 0.45:
            label = "ITEM_AMOUNT"
        elif any(c.isalpha() for c in text) and 0.25 <= y_norm <= 0.85 and "vnd" not in compact:
            label = "ITEM_NAME"
        else:
            label = "OTHER"
        labels.append(LABEL_TO_ID[label])
    return labels
