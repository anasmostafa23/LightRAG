"""
Evaluation pipeline configuration.

Two backends are supported:
  "ollama"  — everything runs locally via Ollama (original local setup)
  "gemini"  — Google Gemini API (free tier, works on Colab CPU, no GPU needed)

Edit the values below or override any field via EVAL_* environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalConfig:
    # ── Backend selection ──────────────────────────────────────────────────
    # "ollama"  → local Ollama server
    # "gemini"  → Google Gemini API (get a free key at aistudio.google.com)
    backend: Literal["ollama", "gemini"] = "ollama"

    # ── Paths ──────────────────────────────────────────────────────────────
    working_dir: str = "./eval/rag_storage"   # LightRAG index lives here
    corpus_dir: str = "./eval/corpus"          # Drop your .txt/.md docs here
    questions_file: str = "./eval/questions.json"
    results_dir: str = "./eval/results"

    # ── Ollama settings (used when backend="ollama") ────────────────────────
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"            # Needs ≥ 32k context
    embedding_model: str = "bge-m3:latest"    # Must stay consistent after indexing
    embedding_dim: int = 1024                  # bge-m3 output dim

    # ── Gemini settings (used when backend="gemini") ───────────────────────
    # Free key at: https://aistudio.google.com/app/apikey
    gemini_api_key: str = ""                   # or set GEMINI_API_KEY env var
    gemini_llm_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "models/text-embedding-004"
    gemini_embed_dim: int = 768                # text-embedding-004 output dim

    # ── Judge LLM ──────────────────────────────────────────────────────────
    # Uses the same backend as above.
    # Ollama:  judge_model is an Ollama model name (can differ from llm_model)
    # Gemini:  judge_model is a Gemini model name
    judge_model: str = "qwen2.5:14b"          # overridden automatically for gemini

    # ── Query settings ─────────────────────────────────────────────────────
    modes: list = field(
        default_factory=lambda: ["naive", "local", "global", "hybrid", "mix"]
    )
    top_k: int = 60
    chunk_top_k: int = 10

    # ── Pipeline behaviour ─────────────────────────────────────────────────
    skip_ingestion: bool = False   # True = reuse existing index
    force_reindex: bool = False    # True = wipe + rebuild from corpus
    max_parallel_insert: int = 2

    def effective_embedding_dim(self) -> int:
        return self.gemini_embed_dim if self.backend == "gemini" else self.embedding_dim

    def effective_judge_model(self) -> str:
        if self.backend == "gemini" and self.judge_model == "qwen2.5:14b":
            return self.gemini_llm_model   # default for gemini
        return self.judge_model


def load_config() -> EvalConfig:
    """Build config from defaults, overridden by EVAL_* environment variables."""
    cfg = EvalConfig()

    overrides = {
        "backend":              "EVAL_BACKEND",
        "working_dir":          "EVAL_WORKING_DIR",
        "corpus_dir":           "EVAL_CORPUS_DIR",
        "questions_file":       "EVAL_QUESTIONS_FILE",
        "results_dir":          "EVAL_RESULTS_DIR",
        "ollama_host":          "EVAL_OLLAMA_HOST",
        "llm_model":            "EVAL_LLM_MODEL",
        "embedding_model":      "EVAL_EMBEDDING_MODEL",
        "embedding_dim":        "EVAL_EMBEDDING_DIM",
        "gemini_api_key":       "GEMINI_API_KEY",
        "gemini_llm_model":     "EVAL_GEMINI_LLM_MODEL",
        "gemini_embed_model":   "EVAL_GEMINI_EMBED_MODEL",
        "gemini_embed_dim":     "EVAL_GEMINI_EMBED_DIM",
        "judge_model":          "EVAL_JUDGE_MODEL",
        "top_k":                "EVAL_TOP_K",
        "chunk_top_k":          "EVAL_CHUNK_TOP_K",
        "skip_ingestion":       "EVAL_SKIP_INGESTION",
        "force_reindex":        "EVAL_FORCE_REINDEX",
        "max_parallel_insert":  "EVAL_MAX_PARALLEL_INSERT",
    }

    int_fields = {
        "embedding_dim", "gemini_embed_dim", "top_k",
        "chunk_top_k", "max_parallel_insert",
    }
    bool_fields = {"skip_ingestion", "force_reindex"}

    for attr, env_var in overrides.items():
        val = os.getenv(env_var)
        if val is not None:
            if attr in int_fields:
                setattr(cfg, attr, int(val))
            elif attr in bool_fields:
                setattr(cfg, attr, val.lower() in ("1", "true", "yes"))
            else:
                setattr(cfg, attr, val)

    modes_env = os.getenv("EVAL_MODES")
    if modes_env:
        cfg.modes = [m.strip() for m in modes_env.split(",")]

    return cfg
