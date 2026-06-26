from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any

import torch

from pipeline.core.gcn_classifier import InvoiceGCN, build_edge_index, build_features, classify_nodes, infer_gcn_hparams_from_state
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.postprocess import build_invoice_json
from pipeline.core.schema import LABEL_MAP, LABEL_TO_ID, OCRNode


class GCNPipelineService:
    def _infer_shape_from_checkpoint(self, checkpoint_path: str) -> tuple[int, int, int, int]:
        state = torch.load(checkpoint_path, map_location="cpu")

        in_channels = None
        out_channels = None
        for key, tensor in state.items():
            if key.endswith("conv1.lin.weight"):
                in_channels = int(tensor.shape[1])
            if key.endswith("conv2.bias"):
                out_channels = int(tensor.shape[0])
            elif key.endswith("conv2.lin.weight"):
                out_channels = int(tensor.shape[0])

        if in_channels is None or out_channels is None:
            raise RuntimeError(f"Cannot infer model shape from checkpoint: {checkpoint_path}")
        return infer_gcn_hparams_from_state(state, in_channels, out_channels)

    def _classify_nodes_with_checkpoint(
        self,
        nodes: list[OCRNode],
        edges: list[tuple[int, int, float]],
        checkpoint_path: str,
    ) -> tuple[list[int], list[float]]:
        if not nodes:
            return [], []

        state = torch.load(checkpoint_path, map_location="cpu")
        in_channels, hidden_channels, out_channels, num_layers = self._infer_shape_from_checkpoint(checkpoint_path)
        model = InvoiceGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=0.0,
        )
        model.load_state_dict(state)
        requested_device = os.environ.get("GCN_DEVICE", "auto").strip().lower()
        device = torch.device(requested_device if requested_device and requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        model.eval()

        x = build_features(nodes).to(device)
        if x.shape[1] > in_channels:
            x = x[:, :in_channels]
        elif x.shape[1] < in_channels:
            pad = torch.zeros((x.shape[0], in_channels - x.shape[1]), dtype=x.dtype, device=device)
            x = torch.cat([x, pad], dim=1)
        if "nodeonly" in Path(checkpoint_path).stem.lower():
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = build_edge_index(edges).to(device)
        with torch.no_grad():
            logits = model(x, edge_index)
            temperature = max(float(os.environ.get("GCN_INFER_TEMPERATURE", "1.0")), 1e-6)
            probs = torch.softmax(logits / temperature, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        return [int(v) for v in pred.tolist()], [float(v) for v in confidence.tolist()]

    def _merge_checkpoint_with_rules(
        self,
        nodes: list[OCRNode],
        rule_labels: list[int],
        checkpoint_labels: list[int],
        checkpoint_confidences: list[float],
    ) -> list[int]:
        other_id = next((idx for idx, name in LABEL_MAP.items() if name == "OTHER"), 0)
        merged: list[int] = []
        for node, rule_label, gcn_label, confidence in zip(nodes, rule_labels, checkpoint_labels, checkpoint_confidences):
            rule_label = self._rule_label_override(node.text, rule_label)

            if not self._gcn_label_is_compatible(node.text, gcn_label):
                merged.append(rule_label)
                continue

            if self._rule_label_is_locked(node.text, rule_label):
                merged.append(rule_label)
                continue

            if gcn_label == other_id and rule_label != other_id:
                merged.append(rule_label)
            elif rule_label == other_id and gcn_label != other_id:
                merged.append(gcn_label if confidence >= 0.78 else rule_label)
            elif rule_label != other_id and gcn_label != rule_label:
                merged.append(gcn_label if confidence >= 0.86 else rule_label)
            else:
                merged.append(gcn_label)
        return merged

    def _norm_text(self, text: str) -> str:
        import unicodedata

        text = str(text or "").lower().replace("đ", "d")
        text = "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        return re.sub(r"\s+", " ", text).strip()

    def _compact_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", self._norm_text(text))

    def _has_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        norm = self._norm_text(text)
        compact = self._compact_text(text)
        return any(keyword in norm or self._compact_text(keyword) in compact for keyword in keywords)

    def _looks_money(self, text: str) -> bool:
        raw = str(text or "").strip()
        digits = re.sub(r"\D", "", raw)
        return bool(digits) and bool(re.fullmatch(r"[-+]?\d[\d.,]*", raw)) and (len(digits) >= 3 or "," in raw or "." in raw)

    def _rule_label_override(self, text: str, rule_label: int) -> int:
        label = LABEL_MAP.get(rule_label, "OTHER")
        has_date = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text or ""))
        has_time = bool(re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text or ""))

        if self._has_any(text, ("hoa don", "hóa đơn", "bien lai", "biên lai", "ten hang", "tên hàng", "tan hang", "tân hàng", "don gia", "đơn giá", "gia ban", "giá bán", "thanh tien", "thành tiền", "dvt", " sl ")):
            return LABEL_TO_ID["OTHER"]
        if self._has_any(text, ("xin cam on", "xin cảm ơn", "hen gap", "hẹn gặp", "website", "tax invoice", "tong dai", "tổng đài", "gop y", "góp ý", "khieu nai", "khiếu nại", "ban sao hd", "hddt")):
            return LABEL_TO_ID["OTHER"]
        if self._has_any(text, ("gia dang ky", "giá đăng ký", "gia dang ky", "giá đăng kí", "gia ban", "giá bán")):
            return LABEL_TO_ID["OTHER"]
        if self._has_any(text, ("tien mat", "tiền mặt", "vnd", "momo", "visa", "mastercard", "chuyen khoan", "chuyển khoản")):
            return LABEL_TO_ID["PAYMENT_METHOD"]
        if self._has_any(text, ("thu ngan", "thu ngân", "cashier", "nhan vien", "nhân viên")):
            return LABEL_TO_ID["CASHIER"]
        if has_date:
            return LABEL_TO_ID["DATE"]
        if has_time:
            return LABEL_TO_ID["TIME"]
        if self._has_any(text, ("tong so", "tổng số", "tong cong", "tổng cộng", "tong tien", "tổng tiền", "tong thanh tien", "tổng thành tiền", "phai thanh toan", "phải thanh toán", "tien khach", "tiền khách", "khach tra", "khách trả", "tien tra", "tiền trả", "tien thoi", "tiền thối", "thoi lai", "thối lại")):
            return LABEL_TO_ID["OTHER"] if not self._looks_money(text) else rule_label

        if label in {"DATE", "TIME"}:
            return LABEL_TO_ID["OTHER"]
        if label in {"ITEM_NAME"} and self._has_any(text, ("tong", "tổng", "phai thanh toan", "phải thanh toán", "gia dang ky", "giá đăng ký", "gia ban", "giá bán")):
            return LABEL_TO_ID["OTHER"]
        return rule_label

    def _rule_label_is_locked(self, text: str, rule_label: int) -> bool:
        label = LABEL_MAP.get(rule_label, "OTHER")
        if label in {"MERCHANT_NAME", "MERCHANT_ADDRESS", "MERCHANT_PHONE", "DATE", "TIME", "CASHIER", "PAYMENT_METHOD", "TAX_CODE", "INVOICE_ID"}:
            return True
        if self._has_any(
            text,
            (
                "hoa don", "hóa đơn", "tong", "tổng", "vnd",
                "ten hang", "tên hàng", "tan hang", "tân hàng", "don gia", "đơn giá",
                "gia ban", "giá bán", "gia dang ky", "giá đăng ký", "gia dang ki", "giá đăng kí",
                "thanh tien", "thành tiền", "dvt", " sl ",
                "thu ngan", "thu ngân", "xin cam on", "xin cảm ơn",
                "tong dai", "tổng đài", "gop y", "góp ý", "khieu nai", "khiếu nại",
            ),
        ):
            return True
        return False

    def _gcn_label_is_compatible(self, text: str, gcn_label: int) -> bool:
        label = LABEL_MAP.get(gcn_label, "OTHER")
        has_digit = bool(re.search(r"\d", text or ""))
        has_date = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text or ""))
        has_time = bool(re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text or ""))
        has_alpha = any(ch.isalpha() for ch in str(text or ""))

        if label == "OTHER":
            return True
        if label == "DATE":
            return has_date or (has_digit and self._has_any(text, ("ngay", "ngày")))
        if label == "TIME":
            return has_time
        if label == "PAYMENT_METHOD":
            return self._has_any(text, ("vnd", "tien mat", "tiền mặt", "cash", "momo", "visa", "mastercard", "chuyen khoan", "chuyển khoản"))
        if label in {"ITEM_UNIT_PRICE", "ITEM_AMOUNT", "TOTAL_AMOUNT", "SUBTOTAL", "SERVICE_FEE", "DISCOUNT", "TAX_AMOUNT"}:
            if self._has_any(text, ("tong dai", "tổng đài", "gop y", "góp ý", "khieu nai", "khiếu nại", "hotline", "hddt")):
                return False
            return self._looks_money(text)
        if label == "ITEM_QTY":
            return bool(re.fullmatch(r"\s*\d{1,2}(?:[.,]\d)?\s*", str(text or "")))
        if label in {"MERCHANT_PHONE", "TAX_CODE", "INVOICE_ID"}:
            return has_digit
        if label in {"MERCHANT_NAME", "MERCHANT_ADDRESS", "CASHIER", "ITEM_NAME"}:
            return has_alpha
        return True

    def infer(self, nodes: list[OCRNode], checkpoint_path: str | None = None) -> dict[str, Any]:
        edges = build_graph_edges(nodes)
        x = build_features(nodes)
        feature_names = [
            "text_len",
            "has_digit",
            "has_money_token",
            "cx_norm",
            "cy_norm",
            "w_norm",
            "h_norm",
            "ocr_score",
            "looks_money",
            "has_total_kw",
            "has_item_header_kw",
            "has_date",
            "has_time",
            "has_phone",
            "has_tax_kw",
            "has_invoice_kw",
            "has_payment_kw",
            "is_top",
            "is_middle",
            "is_bottom",
            "is_left",
            "is_center",
            "is_right",
            "node_order",
            "has_alpha",
            "has_vnd",
            "has_subtotal_kw",
            "has_discount_kw",
            "has_service_kw",
            "has_cashier_kw",
            "has_address_kw",
            "has_receipt_title_kw",
            "has_footer_kw",
            "has_unit_word",
            "qty_candidate",
            "long_numeric",
            "digit_count_norm",
            "numeric_value_norm",
            "x_band_0",
            "x_band_1",
            "x_band_2",
            "x_band_3",
            "x_band_4",
            "line_idx_norm",
            "pos_in_line_norm",
            "line_len_norm",
            "is_first_in_line",
            "is_last_in_line",
            "money_count_in_line",
            "is_rightmost_money_in_line",
            "is_header_line",
            "in_item_region",
            "in_total_region",
            "line_has_total_kw",
            "line_has_tax_kw",
            "line_has_subtotal_kw",
            "line_has_payment_kw",
            "line_has_invoice_kw",
            "line_has_cashier_kw",
            "line_has_phone_kw",
            "line_has_date",
            "line_has_time",
        ]
        rule_labels = classify_nodes(nodes, edges)
        if checkpoint_path:
            raw_gcn_labels, gcn_confidences = self._classify_nodes_with_checkpoint(nodes, edges, checkpoint_path)
            labels = self._merge_checkpoint_with_rules(nodes, rule_labels, raw_gcn_labels, gcn_confidences)
            classifier_mode = "trained_gcn_checkpoint_hybrid_safe"
        else:
            labels = rule_labels
            raw_gcn_labels = labels
            gcn_confidences = [1.0 for _ in labels]
            classifier_mode = "rule_based_fallback"
        extracted = build_invoice_json(nodes, labels)
        rule_extracted = build_invoice_json(nodes, rule_labels)

        node_features = []
        for i, n in enumerate(nodes):
            neighbor_edges = [
                {"target": int(dst), "weight": float(weight)}
                for src, dst, weight in edges
                if src == i
            ]
            node_features.append(
                {
                    "node_id": i,
                    "text": n.text,
                    "x_center": float(n.cx),
                    "y_center": float(n.cy),
                    "width": float(n.w),
                    "height": float(n.h),
                    "confidence": float(n.score),
                    "rule_label": LABEL_MAP.get(rule_labels[i], "OTHER"),
                    "raw_gcn_label": LABEL_MAP.get(raw_gcn_labels[i], "OTHER"),
                    "gcn_confidence": float(gcn_confidences[i]),
                    "predicted_label": LABEL_MAP.get(labels[i], "OTHER"),
                    "changed_by_gcn": LABEL_MAP.get(rule_labels[i], "OTHER") != LABEL_MAP.get(labels[i], "OTHER"),
                    "neighbors": neighbor_edges,
                    "feature_vector": [float(v) for v in x[i].tolist()],
                }
            )

        graph_info = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "feature_names": feature_names,
            "edges": [
                {"source": int(src), "target": int(dst), "weight": float(weight)}
                for src, dst, weight in edges
            ],
        }

        return {
            "flow": "ocr -> graph -> node_features -> node_classification -> invoice_json",
            "classifier_mode": classifier_mode,
            "checkpoint_path": checkpoint_path,
            "graph": graph_info,
            "rule_baseline": {
                "mode": "rule_based_context_classifier",
                **rule_extracted,
            },
            "node_features": node_features,
            **extracted,
        }

    def save_result(self, result: dict[str, Any], output_json_path: str) -> str:
        out = Path(output_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
