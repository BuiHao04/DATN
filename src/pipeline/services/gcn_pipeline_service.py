from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from pipeline.core.gcn_classifier import InvoiceGCN, build_edge_index, build_features, classify_nodes
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.postprocess import build_invoice_json
from pipeline.core.schema import LABEL_MAP, OCRNode


class GCNPipelineService:
    def _infer_shape_from_checkpoint(self, checkpoint_path: str) -> tuple[int, int]:
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
        return in_channels, out_channels

    def _classify_nodes_with_checkpoint(
        self,
        nodes: list[OCRNode],
        edges: list[tuple[int, int, float]],
        checkpoint_path: str,
    ) -> list[int]:
        if not nodes:
            return []

        in_channels, out_channels = self._infer_shape_from_checkpoint(checkpoint_path)
        model = InvoiceGCN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        x = build_features(nodes)
        edge_index = build_edge_index(edges)
        with torch.no_grad():
            logits = model(x, edge_index)
            pred = torch.argmax(logits, dim=1)
        return [int(v) for v in pred.tolist()]

    def infer(self, nodes: list[OCRNode], checkpoint_path: str | None = None) -> dict[str, Any]:
        edges = build_graph_edges(nodes)
        x = build_features(nodes)
        feature_names = ["text_len", "has_digit", "has_money_token", "cx_norm", "cy_norm", "w_norm", "h_norm", "ocr_score"]
        rule_labels = classify_nodes(nodes, edges)
        if checkpoint_path:
            labels = self._classify_nodes_with_checkpoint(nodes, edges, checkpoint_path)
            classifier_mode = "trained_gcn_checkpoint"
        else:
            labels = rule_labels
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
