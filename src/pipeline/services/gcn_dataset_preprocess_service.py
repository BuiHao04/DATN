from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from pipeline.core.gcn_classifier import build_features
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.schema import LABEL_MAP, LABEL_TO_ID, OCRNode

LABEL_ALIASES = {
    "PRODUCT_NAME": "ITEM_NAME",
    "UNIT_PRICE": "ITEM_UNIT_PRICE",
}

ROW_NUMBER_KEY = "__csv_row_number"
INVALID_LABEL_SAMPLE_LIMIT = 20
MALFORMED_ROW_SAMPLE_LIMIT = 20


class GCNDatasetPreprocessService:
    """Build GCN training dataset JSON from tabular CSV annotations."""

    def preprocess_csv(
        self,
        input_csv_path: str,
        output_json_path: str,
        doc_id_col: str = "doc_id",
        text_col: str = "text",
        label_col: str = "label",
        score_col: str = "score",
        x1_col: str = "x1",
        y1_col: str = "y1",
        x2_col: str = "x2",
        y2_col: str = "y2",
        same_line_ratio: float = 1.2,
        near_threshold: float = 250.0,
        min_nodes_per_graph: int = 1,
    ) -> str:
        rows = self._read_csv(input_csv_path)
        samples: list[dict[str, Any]] = []

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            doc_id = (row.get(doc_id_col) or "").strip()
            if not doc_id:
                raise ValueError(f"Missing doc id in column '{doc_id_col}'")
            grouped.setdefault(doc_id, []).append(row)

        invalid_label_rows: list[dict[str, Any]] = []
        malformed_rows: list[dict[str, Any]] = []
        built_nodes = 0

        for doc_id, doc_rows in grouped.items():
            nodes, labels, doc_invalid_label_rows, doc_malformed_rows = self._build_nodes_and_labels(
                doc_id=doc_id,
                doc_rows=doc_rows,
                text_col=text_col,
                label_col=label_col,
                score_col=score_col,
                x1_col=x1_col,
                y1_col=y1_col,
                x2_col=x2_col,
                y2_col=y2_col,
            )
            invalid_label_rows.extend(doc_invalid_label_rows)
            malformed_rows.extend(doc_malformed_rows)
            if len(nodes) < min_nodes_per_graph:
                continue
            built_nodes += len(nodes)

            x = build_features(nodes).tolist()
            edges = build_graph_edges(
                nodes,
                same_line_ratio=same_line_ratio,
                near_threshold=near_threshold,
            )
            edge_index = [[], []]
            for src, dst, _ in edges:
                edge_index[0].append(src)
                edge_index[1].append(dst)

            samples.append(
                {
                    "doc_id": doc_id,
                    "x": x,
                    "edge_index": edge_index,
                    "y": labels,
                }
            )

        out = {
            "meta": {
                "source_csv": str(input_csv_path),
                "num_graphs": len(samples),
                "num_nodes": built_nodes,
                "skipped_invalid_label_rows": len(invalid_label_rows),
                "invalid_label_rows_sample": invalid_label_rows[:INVALID_LABEL_SAMPLE_LIMIT],
                "skipped_malformed_rows": len(malformed_rows),
                "malformed_rows_sample": malformed_rows[:MALFORMED_ROW_SAMPLE_LIMIT],
                "label_map": LABEL_TO_ID,
            },
            "samples": samples,
        }

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if invalid_label_rows:
            logger.warning(
                "Có {} dòng label sai, đã bỏ qua. Sample lỗi đầu tiên: {}",
                len(invalid_label_rows),
                invalid_label_rows[:5],
            )
        if malformed_rows:
            logger.warning(
                "Có {} dòng thiếu/sai dữ liệu node, đã bỏ qua. Sample lỗi đầu tiên: {}",
                len(malformed_rows),
                malformed_rows[:5],
            )
        logger.info(
            "Saved GCN dataset JSON: {} (graphs={}, nodes={}, skipped_invalid_labels={}, skipped_malformed_rows={})",
            output_path,
            len(samples),
            built_nodes,
            len(invalid_label_rows),
            len(malformed_rows),
        )
        return str(output_path)

    def _read_csv(self, input_csv_path: str) -> list[dict[str, str]]:
        path = Path(input_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {input_csv_path}")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                row[ROW_NUMBER_KEY] = str(row_number)
                rows.append(row)
            return rows

    def _build_nodes_and_labels(
        self,
        doc_id: str,
        doc_rows: list[dict[str, str]],
        text_col: str,
        label_col: str,
        score_col: str,
        x1_col: str,
        y1_col: str,
        x2_col: str,
        y2_col: str,
    ) -> tuple[list[OCRNode], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[OCRNode] = []
        labels: list[int] = []
        invalid_label_rows: list[dict[str, Any]] = []
        malformed_rows: list[dict[str, Any]] = []

        for row in doc_rows:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue

            try:
                label_id = self._parse_label(row.get(label_col), label_col)
            except ValueError as exc:
                bad_row = {
                    "row_number": self._to_int(row.get(ROW_NUMBER_KEY)),
                    "doc_id": doc_id,
                    "text": self._short_text(text),
                    "label": row.get(label_col),
                    "reason": str(exc),
                }
                invalid_label_rows.append(bad_row)
                logger.warning(
                    "Skip invalid label row_number={} doc_id={} text={!r} label={!r}: {}",
                    bad_row["row_number"],
                    doc_id,
                    bad_row["text"],
                    bad_row["label"],
                    bad_row["reason"],
                )
                continue

            try:
                x1 = self._to_float(row.get(x1_col), x1_col)
                y1 = self._to_float(row.get(y1_col), y1_col)
                x2 = self._to_float(row.get(x2_col), x2_col)
                y2 = self._to_float(row.get(y2_col), y2_col)
                score = self._to_float(row.get(score_col, "1.0"), score_col, default=1.0)
            except ValueError as exc:
                bad_row = {
                    "row_number": self._to_int(row.get(ROW_NUMBER_KEY)),
                    "doc_id": doc_id,
                    "text": self._short_text(text),
                    "label": row.get(label_col),
                    "reason": str(exc),
                }
                malformed_rows.append(bad_row)
                logger.warning(
                    "Skip malformed node row_number={} doc_id={} text={!r} label={!r}: {}",
                    bad_row["row_number"],
                    doc_id,
                    bad_row["text"],
                    bad_row["label"],
                    bad_row["reason"],
                )
                continue

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            w = max(x2 - x1, 1e-6)
            h = max(y2 - y1, 1e-6)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            node = OCRNode(
                text=text,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
            )

            nodes.append(node)
            labels.append(label_id)

        return nodes, labels, invalid_label_rows, malformed_rows

    def _parse_label(self, raw: str | None, label_col: str) -> int:
        if raw is None:
            raise ValueError(f"Missing label in column '{label_col}'")
        value = raw.strip()
        if not value:
            raise ValueError(f"Empty label in column '{label_col}'")

        if value.isdigit():
            label_id = int(value)
            if label_id not in LABEL_MAP:
                valid = ", ".join(str(k) for k in sorted(LABEL_MAP.keys()))
                raise ValueError(f"Unknown numeric label '{value}'. Valid label ids: {valid}")
            return label_id

        norm = value.upper()
        if norm in LABEL_ALIASES:
            norm = LABEL_ALIASES[norm]
        if norm not in LABEL_TO_ID:
            valid = ", ".join(sorted(LABEL_TO_ID.keys()))
            raise ValueError(f"Unknown label '{value}'. Valid labels: {valid}")
        return LABEL_TO_ID[norm]

    def _to_float(self, raw: str | None, col_name: str, default: float | None = None) -> float:
        if raw is None or raw.strip() == "":
            if default is not None:
                return default
            raise ValueError(f"Missing numeric value in column '{col_name}'")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid float at column '{col_name}': {raw}") from exc

    def _to_int(self, raw: str | None) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _short_text(self, value: str, limit: int = 120) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."
