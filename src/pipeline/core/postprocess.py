from __future__ import annotations

from typing import Any, Dict, List

from pipeline.core.schema import LABEL_MAP, OCRNode


def build_invoice_json(nodes: List[OCRNode], labels: List[int]) -> Dict[str, Any]:
    fields: Dict[str, List[str]] = {
        "DATE": [],
        "TAX_CODE": [],
        "TOTAL_AMOUNT": [],
        "PRODUCT_NAME": [],
        "UNIT_PRICE": [],
    }

    labeled_nodes = []
    for node, label_id in zip(nodes, labels):
        label_name = LABEL_MAP.get(label_id, "OTHER")
        labeled_nodes.append(
            {
                "text": node.text,
                "label": label_name,
                "bbox": [node.x1, node.y1, node.x2, node.y2],
                "score": node.score,
            }
        )
        if label_name in fields:
            fields[label_name].append(node.text)

    return {
        "invoice": {
            "date": fields["DATE"][0] if fields["DATE"] else None,
            "tax_code": fields["TAX_CODE"][0] if fields["TAX_CODE"] else None,
            "total_amount": fields["TOTAL_AMOUNT"][0] if fields["TOTAL_AMOUNT"] else None,
            "product_name": fields["PRODUCT_NAME"],
            "unit_price": fields["UNIT_PRICE"],
        },
        "nodes": labeled_nodes,
    }
