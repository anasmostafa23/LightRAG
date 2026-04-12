"""
Query runner.
Runs each question against every requested LightRAG mode.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass, field

from lightrag import LightRAG, QueryParam
from lightrag.utils import logger

from .config import EvalConfig


@dataclass
class QueryResult:
    question_id: str
    question: str
    mode: str
    answer: str
    latency_s: float
    error: str = ""


async def _query_one(
    rag: LightRAG,
    question_id: str,
    question: str,
    mode: str,
    cfg: EvalConfig,
) -> QueryResult:
    param = QueryParam(
        mode=mode,
        top_k=cfg.top_k,
        chunk_top_k=cfg.chunk_top_k,
        max_total_tokens=cfg.max_total_tokens,
        stream=False,
        enable_rerank=cfg.enable_rerank,
    )
    t0 = time.perf_counter()
    try:
        answer = await rag.aquery(question, param=param)
        # Guard against streaming generators leaking through
        if inspect.isasyncgen(answer):
            chunks = []
            async for chunk in answer:
                chunks.append(chunk)
            answer = "".join(chunks)
        latency = time.perf_counter() - t0
        return QueryResult(
            question_id=question_id,
            question=question,
            mode=mode,
            answer=str(answer),
            latency_s=round(latency, 2),
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        logger.error(f"Query failed [{mode}] q={question_id}: {exc}")
        return QueryResult(
            question_id=question_id,
            question=question,
            mode=mode,
            answer="",
            latency_s=round(latency, 2),
            error=str(exc),
        )


async def run_queries(
    rag: LightRAG,
    questions: list[dict],
    cfg: EvalConfig,
) -> list[QueryResult]:
    """
    For each question × each mode, call aquery and collect results.
    Questions are processed sequentially to avoid overloading Ollama.
    """
    results: list[QueryResult] = []
    total = len(questions) * len(cfg.modes)
    done = 0

    for q in questions:
        qid = q["id"]
        question = q["question"]
        logger.info(f"[{qid}] Querying across {len(cfg.modes)} mode(s) …")

        for mode in cfg.modes:
            result = await _query_one(rag, qid, question, mode, cfg)
            results.append(result)
            done += 1
            status = "OK" if not result.error else f"ERROR: {result.error[:60]}"
            logger.info(
                f"  [{done}/{total}] mode={mode} latency={result.latency_s}s  {status}"
            )

    return results
