"""
Main evaluation pipeline entry point.

Usage:
    python -m eval.pipeline                          # defaults from config.py
    python -m eval.pipeline --skip-ingestion         # reuse existing index
    python -m eval.pipeline --force-reindex          # wipe + rebuild index
    python -m eval.pipeline --modes local,hybrid     # only run specific modes
    python -m eval.pipeline --questions my_qs.json   # custom questions file
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make sure the repo root is on the path when run as a module
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lightrag.utils import logger

from .config import load_config, EvalConfig
from .ingest import ingest
from .query import run_queries
from .judge import score_all
from .report import save_results, print_summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightRAG evaluation pipeline")
    p.add_argument("--questions", "-q", help="Path to questions JSON file")
    p.add_argument(
        "--modes", "-m",
        help="Comma-separated list of query modes (e.g. local,hybrid,mix)"
    )
    p.add_argument("--skip-ingestion", action="store_true",
                   help="Skip document ingestion; reuse existing index")
    p.add_argument("--force-reindex", action="store_true",
                   help="Wipe the existing index and rebuild from corpus")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip LLM judging — only collect raw answers")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_questions(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}\n"
            "See eval/questions.json for the expected format."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {path}")

    # Validate required fields
    for i, q in enumerate(questions):
        for field in ("id", "question", "ground_truth"):
            if field not in q:
                raise ValueError(f"Question [{i}] is missing required field '{field}'")

    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(cfg: EvalConfig, no_judge: bool = False) -> None:
    questions = _load_questions(cfg.questions_file)
    logger.info(
        f"Loaded {len(questions)} question(s) | modes: {cfg.modes} | "
        f"judge: {'disabled' if no_judge else cfg.judge_model}"
    )

    # 1. Ingest documents (or reuse existing index)
    rag = await ingest(cfg)

    try:
        # 2. Run queries across all modes
        logger.info("Running queries …")
        query_results = await run_queries(rag, questions, cfg)

        # 3. Score answers (optional)
        if no_judge:
            scores = []
            logger.info("Skipping judge scoring (--no-judge).")
        else:
            logger.info("Scoring answers with judge model …")
            scores = await score_all(query_results, questions, cfg)

        # 4. Save results
        csv_path, json_path = save_results(query_results, scores, questions, cfg)
        logger.info(f"Results saved:\n  CSV  → {csv_path}\n  JSON → {json_path}")

        # 5. Print summary table
        if scores:
            print_summary(scores, cfg.modes)

    finally:
        await rag.finalize_storages()


def main() -> None:
    args = _parse_args()
    cfg = load_config()

    # CLI overrides
    if args.questions:
        cfg.questions_file = args.questions
    if args.modes:
        cfg.modes = [m.strip() for m in args.modes.split(",")]
    if args.skip_ingestion:
        cfg.skip_ingestion = True
    if args.force_reindex:
        cfg.force_reindex = True

    asyncio.run(run(cfg, no_judge=args.no_judge))


if __name__ == "__main__":
    main()
