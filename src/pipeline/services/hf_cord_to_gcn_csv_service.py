from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from pipeline.core.schema import LABEL_TO_ID


class HFCordToGcnCsvService:
    """Convert Hugging Face CORD-style datasets to node-level CSV for GCN preprocessing."""

    def convert(
        self,
        dataset_id: str,
        split: str,
        output_csv_path: str,
        limit: int | None = None,
    ) -> str:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("Missing dependency 'datasets'. Install with: pip install datasets") from exc

        ds = load_dataset(dataset_id, split=split)
        rows: list[dict[str, str]] = []
        skipped_docs = 0

        for idx, sample in enumerate(ds):
            if limit is not None and idx >= limit:
                break

            doc_id = self._build_doc_id(sample=sample, idx=idx)
            gt = self._parse_ground_truth(sample)
            word_anns = self._extract_word_annotations(gt)
            if not word_anns:
                skipped_docs += 1
                continue

            for ann in word_anns:
                label = self._map_cord_label(ann["category"])
                if label is None:
                    continue
                rows.append(
                    {
                        "doc_id": doc_id,
                        "text": ann["text"],
                        "label": label,
                        "x1": f"{ann['x1']:.2f}",
                        "y1": f"{ann['y1']:.2f}",
                        "x2": f"{ann['x2']:.2f}",
                        "y2": f"{ann['y2']:.2f}",
                        "score": "1.0",
                    }
                )

        if not rows:
            raise ValueError(
                "No usable annotations found. This CORD variant may not include word-level bbox annotations."
            )

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Saved CORD->GCN CSV: {} (rows={}, skipped_docs={})",
            output_path,
            len(rows),
            skipped_docs,
        )
        logger.info("Supported GCN labels: {}", ", ".join(sorted(LABEL_TO_ID.keys())))
        return str(output_path)

    def _build_doc_id(self, sample: dict[str, Any], idx: int) -> str:
        meta = sample.get("meta") if isinstance(sample, dict) else None
        if isinstance(meta, dict) and "image_id" in meta:
            return f"cord_{meta['image_id']}"
        if "id" in sample:
            return str(sample["id"])
        return f"cord_{idx}"

    def _parse_ground_truth(self, sample: dict[str, Any]) -> dict[str, Any]:
        raw = sample.get("ground_truth")
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _extract_word_annotations(self, gt: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        valid_lines = gt.get("valid_line")
        if isinstance(valid_lines, list):
            for line in valid_lines:
                if not isinstance(line, dict):
                    continue
                category = str(line.get("category", "")).strip()
                words = line.get("words", [])
                if not isinstance(words, list):
                    continue
                for w in words:
                    ann = self._parse_word(word=w, category=category)
                    if ann is not None:
                        out.append(ann)

        gt_parse = gt.get("gt_parse")
        if not out and isinstance(gt_parse, dict):
            # Fallback: no word-level boxes in this variant.
            # Keep empty so caller can decide to skip/raise.
            return []

        return out

    def _parse_word(self, word: Any, category: str) -> dict[str, Any] | None:
        if not isinstance(word, dict):
            return None

        text = str(word.get("text", "")).strip()
        if not text:
            return None

        quad = word.get("quad")
        if not isinstance(quad, dict):
            return None

        try:
            x1 = float(quad["x1"])
            y1 = float(quad["y1"])
            x3 = float(quad.get("x3", quad.get("x2", x1)))
            y3 = float(quad.get("y3", quad.get("y2", y1)))
        except Exception:
            return None

        left = min(x1, x3)
        right = max(x1, x3)
        top = min(y1, y3)
        bottom = max(y1, y3)
        if right <= left or bottom <= top:
            return None

        return {
            "text": text,
            "category": category,
            "x1": left,
            "y1": top,
            "x2": right,
            "y2": bottom,
        }

    def _map_cord_label(self, category: str) -> str | None:
        if not category:
            return "OTHER"

        c = category.lower()
        if "date" in c:
            return "DATE"
        if "tax" in c or "biznum" in c:
            return "TAX_CODE"
        if "total" in c or "subtotal" in c or "change" in c or "tax_price" in c:
            return "TOTAL_AMOUNT"
        if "menu.nm" in c or "item" in c or "sub_nm" in c or c.endswith(".nm"):
            return "PRODUCT_NAME"
        if "price" in c or "unitprice" in c or "menu.price" in c:
            return "UNIT_PRICE"
        return "OTHER"
