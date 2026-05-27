from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from pipeline.core.schema import LABEL_TO_ID, OCRNode

try:
    from torch_geometric.nn import GCNConv
except Exception:
    GCNConv = None


def build_features(nodes: List[OCRNode]) -> torch.Tensor:
    max_x = max((n.x2 for n in nodes), default=1.0)
    max_y = max((n.y2 for n in nodes), default=1.0)
    max_x = max(max_x, 1.0)
    max_y = max(max_y, 1.0)

    features = []
    for n in nodes:
        text = n.text.lower()
        features.append(
            [
                min(len(n.text), 64) / 64.0,
                1.0 if any(c.isdigit() for c in text) else 0.0,
                1.0 if any(k in text for k in [".", ",", "đ", "vnd"]) else 0.0,
                n.cx / max_x,
                n.cy / max_y,
                n.w / max_x,
                n.h / max_y,
                n.score,
            ]
        )
    return torch.tensor(features, dtype=torch.float32)


def build_edge_index(edges: List[Tuple[int, int, float]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    raw = [[src, dst] for src, dst, _ in edges]
    return torch.tensor(raw, dtype=torch.long).t().contiguous()


class InvoiceGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch-geometric is not installed")
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def classify_nodes(nodes: List[OCRNode], edges: List[Tuple[int, int, float]]) -> List[int]:
    if not nodes:
        return []

    neighbor_map: dict[int, List[int]] = {i: [] for i in range(len(nodes))}
    for src, dst, _ in edges:
        neighbor_map[src].append(dst)

    labels: List[int] = []
    for i, node in enumerate(nodes):
        text = node.text.lower()
        neighbor_text = " ".join(nodes[j].text.lower() for j in neighbor_map[i][:8])
        context = f"{text} {neighbor_text}"

        if text.replace("/", "").isdigit() and len(text) in {8, 10, 12, 13, 14} and "mst" in context:
            labels.append(LABEL_TO_ID["TAX_CODE"])
        elif any(c.isdigit() for c in text) and any(k in context for k in ["tổng cộng", "thành tiền", "total"]):
            labels.append(LABEL_TO_ID["TOTAL_AMOUNT"])
        elif any(c.isdigit() for c in text) and ("/" in text or "-" in text) and len(text) >= 8:
            labels.append(LABEL_TO_ID["DATE"])
        elif any(c.isdigit() for c in text) and any(k in context for k in ["đơn giá", "unit", "price", "vnd"]):
            labels.append(LABEL_TO_ID["ITEM_UNIT_PRICE"])
        elif any(c.isalpha() for c in text):
            labels.append(LABEL_TO_ID["ITEM_NAME"])
        else:
            labels.append(LABEL_TO_ID["OTHER"])
    return labels
