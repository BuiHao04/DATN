from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.core.gcn_classifier import build_features, classify_nodes
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.postprocess import build_invoice_json
from pipeline.core.schema import OCRNode


class GCNPipelineService:
    def infer(self, nodes: list[OCRNode]) -> dict[str, Any]:
        edges = build_graph_edges(nodes)
        x = build_features(nodes)
        labels = classify_nodes(nodes, edges)
        extracted = build_invoice_json(nodes, labels)

        node_features = []
        for i, n in enumerate(nodes):
            node_features.append(
                {
                    "node_id": i,
                    "text": n.text,
                    "x_center": float(n.cx),
                    "y_center": float(n.cy),
                    "width": float(n.w),
                    "height": float(n.h),
                    "confidence": float(n.score),
                    "feature_vector": [float(v) for v in x[i].tolist()],
                }
            )

        graph_info = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "edges": [
                {"source": int(src), "target": int(dst), "weight": float(weight)}
                for src, dst, weight in edges
            ],
        }

        return {
            "flow": "ocr -> graph -> node_features -> node_classification -> invoice_json",
            "graph": graph_info,
            "node_features": node_features,
            **extracted,
        }

    def save_result(self, result: dict[str, Any], output_json_path: str) -> str:
        out = Path(output_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
