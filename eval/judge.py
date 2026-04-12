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
You are a STRICT evaluator for Retrieval-Augmented Generation (RAG) systems.
Your primary job is to identify DIFFERENCES in answer quality — you must use the full 1–5 range.
The majority of answers will score 2 or 3, not 4 or 5.
Key rules:
- If the answer says "I cannot find", "I don't have information", "not found in context", or similar refusals → score answer_relevance=1 and completeness=1.
- Score 5 only when the answer is fully correct AND covers every required reasoning step.
- Score 4 only when the answer is mostly correct with only trivial gaps.
- Score 3 for partial answers that get the main idea but miss key details.
- Score 2 for answers that are mostly wrong or heavily incomplete.
- Score 1 for off-topic, refusal, or completely incorrect answers.
You MUST respond with ONLY a JSON object — no markdown fences, no text outside the JSON.
"""

_JUDGE_TEMPLATE = """\
## Question
{question}

## Ground Truth
{ground_truth}

## RAG System Answer
{answer}

## Scoring rubric — use the FULL 1–5 range, most answers do NOT deserve 5

**faithfulness** — Are all factual claims in the answer supported by the ground truth?
  1 = major hallucinations or outright wrong facts
  2 = mostly wrong with a few correct details
  3 = partially correct but contains notable errors or unsupported claims
  4 = mostly correct with only minor inaccuracies
  5 = every claim is fully supported, zero hallucinations

**answer_relevance** — Does the answer directly address what was asked?
  1 = off-topic, refusal ("I cannot find…"), or completely misses the question
  2 = tangentially related but does not answer the actual question
  3 = addresses part of the question but drifts or omits the core ask
  4 = directly answers the question with minor digressions
  5 = precise, focused answer to exactly what was asked

**completeness** — For multi-hop questions, are ALL required reasoning steps/hops covered?
  1 = answers nothing or only the most trivial surface detail
  2 = covers less than half the required information
  3 = covers roughly half of required hops/facts
  4 = covers most required hops/facts with only one small gap
  5 = every required reasoning step and fact is present

**coherence** — Is the answer logically structured and easy to follow?
  1 = incoherent, contradictory, or impossible to understand
  2 = hard to follow, poorly organised
  3 = understandable but lacks clear structure
  4 = clear and well-organised
  5 = perfectly structured, logical flow, easy to follow

## Required output (strict JSON, no extra text, no markdown)
{{"faithfulness": <1-5>, "answer_relevance": <1-5>, "completeness": <1-5>, "coherence": <1-5>, "reasoning": "<one short paragraph explaining the scores>"}}
"""

# Phrases that indicate a soft "no-answer" from the RAG system.
# These are scored 1 on answer_relevance and completeness before calling the judge.
_NO_ANSWER_PATTERNS = [
    "i cannot find",
    "i don't have information",
    "i do not have information",
    "no information available",
    "not found in the provided",
    "not found in context",
    "the provided context does not",
    "based on the available context, i cannot",
    "i'm unable to find",
    "i am unable to find",
    "sorry, i cannot",
    "unfortunately, i cannot",
]


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

def _is_no_answer(answer: str) -> bool:
    """Return True if the answer is a soft refusal from the RAG system."""
    lower = answer.lower()
    return any(pat in lower for pat in _NO_ANSWER_PATTERNS)


async def _score_one(
    result: QueryResult,
    ground_truth: str,
    cfg: EvalConfig,
) -> Score:
    if result.error:
        return _error_score(result.question_id, result.mode, f"query_error: {result.error}")
    if not result.answer.strip():
        return _error_score(result.question_id, result.mode, "empty_answer")

    # Soft "no-answer" detection: score relevance and completeness as 1,
    # but still run the full judge for faithfulness and coherence.
    if _is_no_answer(result.answer):
        logger.warning(
            f"  [{result.question_id}/{result.mode}] Detected soft no-answer — "
            f"forcing answer_relevance=1, completeness=1"
        )
        if cfg.backend in ("openai", "gemini"):
            score = await _score_api(result, ground_truth, cfg)
        else:
            score = await _score_ollama(result, ground_truth, cfg)
        # Override the two metrics that must be 1 for non-answers
        score.answer_relevance = 1.0
        score.completeness = 1.0
        score.overall = round(
            (score.faithfulness + 1.0 + 1.0 + score.coherence) / 4, 3
        )
        return score

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
