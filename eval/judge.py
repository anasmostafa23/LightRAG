"""
LLM-as-judge scorer.
Calls the judge model and scores each answer on four dimensions.

Two backends:
  "ollama"  — calls Ollama directly via the ollama Python client
  "gemini"  — calls Gemini via its OpenAI-compatible endpoint (free tier)
"""

import json
import re
from dataclasses import dataclass

from lightrag.utils import logger

from .config import EvalConfig
from .query import QueryResult


# ── Score schema ──────────────────────────────────────────────────────────────

@dataclass
class Score:
    question_id: str
    mode: str
    faithfulness: float       # Is the answer grounded in retrievable facts?   1-5
    answer_relevance: float   # Does it directly address the question?          1-5
    completeness: float       # Are all reasoning hops covered?                 1-5
    coherence: float          # Is the reasoning clear and logically sound?     1-5
    overall: float            # Mean of the four above
    judge_reasoning: str = ""
    error: str = ""


# ── Prompt ────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Evaluate answers against a reference ground truth.
You MUST respond with ONLY a JSON object — no markdown fences, no text outside the JSON.
"""

_JUDGE_TEMPLATE = """\
## Question
{question}

## Ground Truth
{ground_truth}

## RAG System Answer
{answer}

## Evaluation criteria (score each 1–5, where 5 is best)

- faithfulness: Every factual claim is supported by the ground truth or well-known facts. Penalise hallucinations.
- answer_relevance: The answer directly addresses what the question asks. Penalise off-topic responses.
- completeness: For multi-hop questions, ALL required reasoning steps/hops are covered. Penalise partial answers.
- coherence: The answer is logically structured and easy to follow.

## Required output (strict JSON, no extra text, no markdown)
{{"faithfulness": <1-5>, "answer_relevance": <1-5>, "completeness": <1-5>, "coherence": <1-5>, "reasoning": "<one short paragraph>"}}
"""


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in judge output:\n{text[:500]}")


def _build_score(question_id: str, mode: str, data: dict) -> Score:
    vals = [
        float(data.get("faithfulness", 0)),
        float(data.get("answer_relevance", 0)),
        float(data.get("completeness", 0)),
        float(data.get("coherence", 0)),
    ]
    return Score(
        question_id=question_id,
        mode=mode,
        faithfulness=vals[0],
        answer_relevance=vals[1],
        completeness=vals[2],
        coherence=vals[3],
        overall=round(sum(vals) / len(vals), 3),
        judge_reasoning=data.get("reasoning", ""),
    )


def _make_messages(question: str, ground_truth: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _JUDGE_TEMPLATE.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
            ),
        },
    ]


def _error_score(question_id: str, mode: str, error: str) -> Score:
    return Score(
        question_id=question_id,
        mode=mode,
        faithfulness=0, answer_relevance=0,
        completeness=0, coherence=0, overall=0,
        error=error,
    )


# ── Ollama judge ──────────────────────────────────────────────────────────────

async def _score_ollama(
    result: QueryResult,
    ground_truth: str,
    cfg: EvalConfig,
) -> Score:
    import ollama

    client = ollama.AsyncClient(host=cfg.ollama_host)
    try:
        response = await client.chat(
            model=cfg.effective_judge_model(),
            messages=_make_messages(result.question, ground_truth, result.answer),
            options={"temperature": 0},
        )
        raw = response["message"]["content"]
        data = _extract_json(raw)
        return _build_score(result.question_id, result.mode, data)
    except Exception as exc:
        return _error_score(result.question_id, result.mode, str(exc))
    finally:
        try:
            await client._client.aclose()
        except Exception:
            pass


# ── OpenAI / Gemini judge (shared — both use the OpenAI client interface) ──────
# Gemini exposes an OpenAI-compatible endpoint, so one function covers both.

_BACKEND_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

async def _score_api(
    result: QueryResult,
    ground_truth: str,
    cfg: EvalConfig,
) -> Score:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=cfg.effective_api_key(),
        base_url=_BACKEND_BASE_URLS[cfg.backend],
    )
    try:
        response = await client.chat.completions.create(
            model=cfg.effective_judge_model(),
            messages=_make_messages(result.question, ground_truth, result.answer),
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        data = _extract_json(raw)
        return _build_score(result.question_id, result.mode, data)
    except Exception as exc:
        return _error_score(result.question_id, result.mode, str(exc))
    finally:
        await client.close()


# ── Dispatcher ────────────────────────────────────────────────────────────────

async def _score_one(
    result: QueryResult,
    ground_truth: str,
    cfg: EvalConfig,
) -> Score:
    if result.error:
        return _error_score(result.question_id, result.mode, f"query_error: {result.error}")
    if not result.answer.strip():
        return _error_score(result.question_id, result.mode, "empty_answer")

    if cfg.backend in ("openai", "gemini"):
        return await _score_api(result, ground_truth, cfg)
    return await _score_ollama(result, ground_truth, cfg)


# ── Public API ────────────────────────────────────────────────────────────────

async def score_all(
    query_results: list[QueryResult],
    questions: list[dict],
    cfg: EvalConfig,
) -> list[Score]:
    """Score every query result using the configured judge model."""
    gt_map = {q["id"]: q["ground_truth"] for q in questions}
    scores: list[Score] = []
    total = len(query_results)

    for i, result in enumerate(query_results, 1):
        ground_truth = gt_map.get(result.question_id, "")
        logger.info(
            f"  Scoring [{i}/{total}] q={result.question_id} mode={result.mode} …"
        )
        score = await _score_one(result, ground_truth, cfg)
        scores.append(score)

    return scores
