"""
Results reporter.
Saves CSV + JSON and prints a summary table to stdout.
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import EvalConfig
from .query import QueryResult
from .judge import Score


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(
    query_results: list[QueryResult],
    scores: list[Score],
    questions: list[dict],
    cfg: EvalConfig,
) -> tuple[str, str]:
    """
    Merge query results + scores, write CSV and JSON.
    Returns (csv_path, json_path).
    """
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    csv_path = results_dir / f"results_{ts}.csv"
    json_path = results_dir / f"results_{ts}.json"

    # Build lookup: (question_id, mode) -> Score
    score_map: dict[tuple[str, str], Score] = {
        (s.question_id, s.mode): s for s in scores
    }

    # Build lookup: question_id -> question dict
    q_map = {q["id"]: q for q in questions}

    rows = []
    for qr in query_results:
        sc = score_map.get((qr.question_id, qr.mode))
        q = q_map.get(qr.question_id, {})
        row = {
            "question_id": qr.question_id,
            "hop_count": q.get("hop_count", ""),
            "mode": qr.mode,
            "question": qr.question,
            "ground_truth": q.get("ground_truth", ""),
            "answer": qr.answer,
            "latency_s": qr.latency_s,
            "query_error": qr.error,
            "faithfulness": sc.faithfulness if sc else "",
            "answer_relevance": sc.answer_relevance if sc else "",
            "completeness": sc.completeness if sc else "",
            "coherence": sc.coherence if sc else "",
            "overall": sc.overall if sc else "",
            "judge_reasoning": sc.judge_reasoning if sc else "",
            "score_error": sc.error if sc else "",
        }
        rows.append(row)

    # ── CSV ──────────────────────────────────────────────────────────────────
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ── JSON ─────────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "config": {
                    "llm_model": cfg.llm_model,
                    "embedding_model": cfg.embedding_model,
                    "judge_model": cfg.judge_model,
                    "modes": cfg.modes,
                    "top_k": cfg.top_k,
                },
                "results": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return str(csv_path), str(json_path)


def print_summary(scores: list[Score], modes: list[str]) -> None:
    """Print a per-mode summary table."""
    from collections import defaultdict

    # Aggregate per mode
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = ["faithfulness", "answer_relevance", "completeness", "coherence", "overall"]

    for s in scores:
        if s.error:
            continue
        for m in metrics:
            sums[s.mode][m].append(getattr(s, m))

    col_w = 17
    header = f"{'Mode':<12}" + "".join(f"{m[:col_w]:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print("EVALUATION SUMMARY (mean scores, 1–5 scale)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for mode in modes:
        d = sums.get(mode, {})
        row = f"{mode:<12}"
        for m in metrics:
            vals = d.get(m, [])
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f"{avg:>{col_w}.2f}"
        print(row)

    print("=" * len(header))
