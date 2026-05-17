from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger


class HFGenericToGcnCsvService:
    """
    Convert a generic Hugging Face dataset split to node-level CSV.
    Output schema:
      doc_id,text,label,x1,y1,x2,y2,score
    """

    def convert(
        self,
        dataset_id: str,
        split: str,
        output_csv_path: str,
        doc_id_field: str = "id",
        text_field: str = "text",
        label_field: str = "label",
        bbox_field: str = "bbox",
        score_field: str | None = None,
        label_map: dict[str, str] | None = None,
        limit: int | None = None,
        streaming: bool = True,
    ) -> str:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("Missing dependency 'datasets'. Install with: pip install datasets") from exc

        ds = load_dataset(dataset_id, split=split, streaming=streaming)
        rows: list[dict[str, str]] = []

        for idx, sample in enumerate(ds):
            if limit is not None and idx >= limit:
                break

            doc_id = self._build_doc_id(sample, doc_id_field, idx)
            text_val = self._get_by_path(sample, text_field)
            label_val = self._get_by_path(sample, label_field)
            bbox_val = self._get_by_path(sample, bbox_field)
            score_val = self._get_by_path(sample, score_field) if score_field else None

            if isinstance(text_val, list):
                rows.extend(
                    self._explode_list_record(
                        doc_id=doc_id,
                        texts=text_val,
                        labels=label_val,
                        bboxes=bbox_val,
                        scores=score_val,
                        label_map=label_map,
                    )
                )
            else:
                one = self._build_one_row(
                    doc_id=doc_id,
                    text=text_val,
                    label=label_val,
                    bbox=bbox_val,
                    score=score_val,
                    label_map=label_map,
                )
                if one:
                    rows.append(one)

        if not rows:
            raise ValueError("No rows generated. Check field names and data format.")

        out = Path(output_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Saved generic HF -> GCN CSV: {} (rows={}, streaming={})",
            out,
            len(rows),
            streaming,
        )
        return str(out)

    def _build_doc_id(self, sample: dict[str, Any], doc_id_field: str, idx: int) -> str:
        val = self._get_by_path(sample, doc_id_field)
        if val is None or str(val).strip() == "":
            return f"doc_{idx}"
        return str(val)

    def _get_by_path(self, obj: Any, path: str | None) -> Any:
        if not path:
            return None
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def _explode_list_record(
        self,
        doc_id: str,
        texts: list[Any],
        labels: Any,
        bboxes: Any,
        scores: Any,
        label_map: dict[str, str] | None,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        n = len(texts)
        for i in range(n):
            text = texts[i]
            label = labels[i] if isinstance(labels, list) and i < len(labels) else labels
            bbox = bboxes[i] if isinstance(bboxes, list) and i < len(bboxes) else bboxes
            score = scores[i] if isinstance(scores, list) and i < len(scores) else scores
            row = self._build_one_row(
                doc_id=doc_id,
                text=text,
                label=label,
                bbox=bbox,
                score=score,
                label_map=label_map,
            )
            if row:
                out.append(row)
        return out

    def _build_one_row(
        self,
        doc_id: str,
        text: Any,
        label: Any,
        bbox: Any,
        score: Any,
        label_map: dict[str, str] | None,
    ) -> dict[str, str] | None:
        txt = "" if text is None else str(text).strip()
        if not txt:
            return None

        box = self._parse_bbox(bbox)
        if box is None:
            return None
        x1, y1, x2, y2 = box

        lab = "" if label is None else str(label).strip()
        if label_map and lab in label_map:
            lab = label_map[lab]
        if not lab:
            lab = "OTHER"

        sc = 1.0
        if score is not None and str(score).strip() != "":
            try:
                sc = float(score)
            except Exception:
                sc = 1.0

        return {
            "doc_id": doc_id,
            "text": txt,
            "label": lab,
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}",
            "y2": f"{y2:.2f}",
            "score": f"{sc:.4f}",
        }

    def _parse_bbox(self, bbox: Any) -> tuple[float, float, float, float] | None:
        if isinstance(bbox, dict):
            try:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                return self._normalize_box(x1, y1, x2, y2)
            except Exception:
                return None

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1 = float(bbox[0])
                y1 = float(bbox[1])
                x2 = float(bbox[2])
                y2 = float(bbox[3])
                return self._normalize_box(x1, y1, x2, y2)
            except Exception:
                return None

        return None

    def _normalize_box(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        left, right = (x1, x2) if x1 <= x2 else (x2, x1)
        top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
        return left, top, right, bottom

    @staticmethod
    def load_label_map(raw: str | None) -> dict[str, str] | None:
        if not raw:
            return None
        p = Path(raw)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return json.loads(raw)
