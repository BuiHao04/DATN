from __future__ import annotations

import math
from typing import List, Tuple

from pipeline.core.schema import OCRNode


def euclidean_distance(a: OCRNode, b: OCRNode) -> float:
    return math.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2)


def build_graph_edges(
    nodes: List[OCRNode],
    same_line_ratio: float = 1.2,
    near_threshold: float = 250.0,
) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            same_line = abs(a.cy - b.cy) < max(a.h, b.h) * same_line_ratio
            dist = euclidean_distance(a, b)
            near = dist < near_threshold
            if same_line or near:
                edges.append((i, j, dist))
                edges.append((j, i, dist))
    return edges
