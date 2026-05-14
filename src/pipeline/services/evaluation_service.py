from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class EvaluationService:
    EVAL_FIELDS = ["date", "tax_code", "total_amount", "seller_name"]

    @staticmethod
    def _norm_text(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def evaluate_invoice_fields(self, pred_invoice: dict[str, Any], gt_invoice: dict[str, Any]) -> dict[str, Any]:
        per_field = {}
        correct = 0

        for field in self.EVAL_FIELDS:
            p = self._norm_text(pred_invoice.get(field))
            g = self._norm_text(gt_invoice.get(field))
            hit = int(p == g and g != "")
            per_field[field] = {
                "pred": pred_invoice.get(field),
                "gt": gt_invoice.get(field),
                "exact_match": bool(hit),
            }
            correct += hit

        total = len(self.EVAL_FIELDS)
        accuracy = correct / total if total else 0.0

        return {
            "fields": per_field,
            "metrics": {
                "field_exact_match_count": correct,
                "field_total": total,
                "field_accuracy": accuracy,
            },
        }

    def evaluate_from_files(self, pred_json_path: str, gt_json_path: str, output_eval_path: str) -> str:
        pred = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))
        gt = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))

        pred_invoice = pred.get("invoice", pred)
        gt_invoice = gt.get("invoice", gt)

        report = self.evaluate_invoice_fields(pred_invoice, gt_invoice)

        out = Path(output_eval_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
