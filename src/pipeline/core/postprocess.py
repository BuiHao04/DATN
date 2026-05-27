from __future__ import annotations

from typing import Any, Dict, List

from pipeline.core.schema import LABEL_MAP, OCRNode


def build_invoice_json(nodes: List[OCRNode], labels: List[int]) -> Dict[str, Any]:
    fields: Dict[str, List[str]] = {
        "MERCHANT_NAME": [],
        "MERCHANT_ADDRESS": [],
        "MERCHANT_PHONE": [],
        "DATE": [],
        "TIME": [],
        "TAX_CODE": [],
        "INVOICE_ID": [],
        "TOTAL_AMOUNT": [],
        "ITEM_NAME": [],
        "ITEM_QTY": [],
        "ITEM_UNIT_PRICE": [],
        "ITEM_AMOUNT": [],
        "SUBTOTAL": [],
        "SERVICE_FEE": [],
        "DISCOUNT": [],
        "TAX_AMOUNT": [],
        "PAYMENT_METHOD": [],
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
            "merchant_name": fields["MERCHANT_NAME"][0] if fields["MERCHANT_NAME"] else None,
            "merchant_address": fields["MERCHANT_ADDRESS"][0] if fields["MERCHANT_ADDRESS"] else None,
            "merchant_phone": fields["MERCHANT_PHONE"][0] if fields["MERCHANT_PHONE"] else None,
            "date": fields["DATE"][0] if fields["DATE"] else None,
            "time": fields["TIME"][0] if fields["TIME"] else None,
            "tax_code": fields["TAX_CODE"][0] if fields["TAX_CODE"] else None,
            "invoice_id": fields["INVOICE_ID"][0] if fields["INVOICE_ID"] else None,
            "subtotal": fields["SUBTOTAL"][0] if fields["SUBTOTAL"] else None,
            "service_fee": fields["SERVICE_FEE"][0] if fields["SERVICE_FEE"] else None,
            "discount": fields["DISCOUNT"][0] if fields["DISCOUNT"] else None,
            "tax_amount": fields["TAX_AMOUNT"][0] if fields["TAX_AMOUNT"] else None,
            "total_amount": fields["TOTAL_AMOUNT"][0] if fields["TOTAL_AMOUNT"] else None,
            "payment_method": fields["PAYMENT_METHOD"][0] if fields["PAYMENT_METHOD"] else None,
            "item_name": fields["ITEM_NAME"],
            "item_qty": fields["ITEM_QTY"],
            "item_unit_price": fields["ITEM_UNIT_PRICE"],
            "item_amount": fields["ITEM_AMOUNT"],
        },
        "nodes": labeled_nodes,
    }
