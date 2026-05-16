from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pipeline.core.gcn_classifier import InvoiceGCN, build_edge_index
from pipeline.core.schema import LABEL_MAP


class GCNEvaluationService:
    def evaluate(
        self,
        dataset_json_path: str,
        checkpoint_path: str,
        output_eval_path: str,
    ) -> str:
        data = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        if not samples:
            raise ValueError("Empty dataset: missing samples")

        in_channels = len(samples[0]["x"][0])
        out_channels = max(max(s["y"]) for s in samples) + 1

        model = InvoiceGCN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        conf = [[0 for _ in range(out_channels)] for _ in range(out_channels)]
        total_nodes = 0
        correct_nodes = 0
        total_loss = 0.0
        total_infer_ms = 0.0

        with torch.no_grad():
            for s in samples:
                x = torch.tensor(s["x"], dtype=torch.float32)
                if "edge_index" in s:
                    edge_index = torch.tensor(s["edge_index"], dtype=torch.long)
                else:
                    edge_index = build_edge_index(s.get("edges", []))
                y = torch.tensor(s["y"], dtype=torch.long)

                t0 = time.perf_counter()
                logits = model(x, edge_index)
                total_infer_ms += (time.perf_counter() - t0) * 1000.0

                loss = F.cross_entropy(logits, y)
                total_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)
                total_nodes += int(y.numel())
                correct_nodes += int((pred == y).sum().item())

                for yi, pi in zip(y.tolist(), pred.tolist()):
                    if 0 <= yi < out_channels and 0 <= pi < out_channels:
                        conf[yi][pi] += 1

        per_class = {}
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0

        for c in range(out_channels):
            tp = conf[c][c]
            fp = sum(conf[r][c] for r in range(out_channels) if r != c)
            fn = sum(conf[c][k] for k in range(out_channels) if k != c)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_class[LABEL_MAP.get(c, str(c))] = {
                "id": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            macro_p += precision
            macro_r += recall
            macro_f1 += f1

        denom = max(out_channels, 1)
        report: dict[str, Any] = {
            "dataset_path": dataset_json_path,
            "checkpoint_path": checkpoint_path,
            "metrics": {
                "accuracy": (correct_nodes / total_nodes) if total_nodes else 0.0,
                "precision_macro": macro_p / denom,
                "recall_macro": macro_r / denom,
                "f1_macro": macro_f1 / denom,
                "loss_avg": total_loss / max(len(samples), 1),
                "inference_time_ms_total": total_infer_ms,
                "inference_time_ms_avg_per_graph": total_infer_ms / max(len(samples), 1),
                "num_graphs": len(samples),
                "num_nodes": total_nodes,
            },
            "per_class": per_class,
            "confusion_matrix": {
                "labels": [LABEL_MAP.get(i, str(i)) for i in range(out_channels)],
                "matrix": conf,
            },
        }

        out = Path(output_eval_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)

