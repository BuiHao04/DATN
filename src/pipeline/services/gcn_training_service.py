from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from pipeline.core.gcn_classifier import InvoiceGCN, build_edge_index


class GCNTrainingService:
    """
    Expect training data as list of graph samples:
    [
      {
        "x": [[...], [...]],
        "edge_index": [[src...], [dst...]],
        "y": [label_id, ...]
      }
    ]
    """

    def train(self, dataset_json_path: str, checkpoint_path: str, epochs: int = 30, lr: float = 1e-3) -> str:
        data = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        if not samples:
            raise ValueError("Empty dataset: missing samples")

        in_channels = len(samples[0]["x"][0])
        out_channels = max(max(s["y"]) for s in samples) + 1

        model = InvoiceGCN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for s in samples:
                x = torch.tensor(s["x"], dtype=torch.float32)
                if "edge_index" in s:
                    edge_index = torch.tensor(s["edge_index"], dtype=torch.long)
                else:
                    edge_index = build_edge_index(s.get("edges", []))
                y = torch.tensor(s["y"], dtype=torch.long)

                optimizer.zero_grad()
                logits = model(x, edge_index)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            logger.info("Epoch {}/{} - loss={:.6f}", epoch, epochs, total_loss / max(len(samples), 1))

        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        logger.info("Saved GCN checkpoint: {}", ckpt)
        return str(ckpt)
