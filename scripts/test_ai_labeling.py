#!/usr/bin/env python3
"""Dry-run AI labeling quality test on a small random batch of invoices.

Calls the running backend's /api/pipeline/labeling-suggest-doc endpoint for N
random doc_ids and reports per-doc strategy, label distribution, unlabeled
counts and sample (text -> label) pairs. Does NOT modify the CSV.

Usage:
    python scripts/test_ai_labeling.py --n 15
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.request as urlreq
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of random docs to test")
    ap.add_argument("--csv", default="data/labeling_stage_b/nodes_to_label.csv")
    ap.add_argument("--base-url", default="http://127.0.0.1:8580")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between docs (avoid rate limit)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv_path = SRC_DIR / args.csv
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    doc_ids = list(dict.fromkeys(r["doc_id"] for r in rows))
    random.seed(args.seed)
    sample_docs = random.sample(doc_ids, min(args.n, len(doc_ids)))
    print(f"Total docs={len(doc_ids)} | testing {len(sample_docs)} random docs\n")

    overall = Counter()
    total_unlabeled = 0
    total_nodes = 0
    total_time = 0.0
    errors_docs = 0

    for i, did in enumerate(sample_docs, 1):
        body = json.dumps({
            "input_csv": args.csv,
            "doc_id": did,
            "only_empty": 0,
            "require_llm": 1,
        }).encode("utf-8")
        req = urlreq.Request(
            f"{args.base_url}/api/pipeline/labeling-suggest-doc",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urlreq.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[{i}/{len(sample_docs)}] {did}  ERROR: {exc}")
            errors_docs += 1
            time.sleep(args.sleep)
            continue
        dt = time.time() - t0
        total_time += dt

        st = d.get("stats", {})
        sug = d.get("suggestions", [])
        dist = Counter(s["label"] for s in sug)
        overall.update(dist)
        total_nodes += len(sug)
        total_unlabeled += int(st.get("llm_unlabeled") or 0)
        if st.get("llm_errors"):
            errors_docs += 1

        print(f"[{i}/{len(sample_docs)}] {did}  {dt:4.1f}s  mode={st.get('strategy_used')} "
              f"nodes={len(sug)} unlabeled={st.get('llm_unlabeled')} err={st.get('llm_errors')[:1]}")
        for s in sug[:6]:
            print(f"      {s['label']:<16} | {s['text'][:48]}")
        time.sleep(args.sleep)

    print("\n=== OVERALL ===")
    print(f"docs_with_errors={errors_docs} | nodes_labeled={total_nodes} | unlabeled={total_unlabeled} | avg_time/doc={total_time/max(1,len(sample_docs)):.1f}s")
    print("label distribution:")
    for k, v in overall.most_common():
        print(f"  {k:<18} {v:5d}  {v*100/max(1,total_nodes):5.1f}%")


if __name__ == "__main__":
    main()
