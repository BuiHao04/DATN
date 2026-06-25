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
    max_near_neighbors: int = 4,
) -> List[Tuple[int, int, float]]:
    edge_map: dict[tuple[int, int], float] = {}

    def add_edge(src: int, dst: int, dist: float) -> None:
        if src == dst:
            return
        key = (src, dst)
        if key not in edge_map or dist < edge_map[key]:
            edge_map[key] = dist

    for i in range(len(nodes)):
        near_candidates: list[tuple[float, int]] = []
        for j in range(len(nodes)):
            if i == j:
                continue
            a = nodes[i]
            b = nodes[j]
            same_line = abs(a.cy - b.cy) < max(a.h, b.h) * same_line_ratio
            dist = euclidean_distance(a, b)
            if same_line:
                add_edge(i, j, dist)
            elif dist < near_threshold:
                near_candidates.append((dist, j))

        for dist, j in sorted(near_candidates)[:max(max_near_neighbors, 0)]:
            add_edge(i, j, dist)

    return [(src, dst, dist) for (src, dst), dist in sorted(edge_map.items())]
