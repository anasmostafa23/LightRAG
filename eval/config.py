"""
Evaluation pipeline configuration.

Three backends are supported:
  "ollama"  — everything runs locally via Ollama
  "openai"  — OpenAI API (gpt-4o-mini + text-embedding-3-small)
  "gemini"  — Google Gemini API (free tier, gemini-2.0-flash)

Edit the values below or override any field via EVAL_* environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalConfig:
    # ── Backend selection ──────────────────────────────────────────────────
    # "ollama"  → local Ollama server
    # "openai"  → OpenAI API  (set OPENAI_API_KEY)
    # "gemini"  → Google Gemini API (set GEMINI_API_KEY, free tier)
    backend: Literal["ollama", "openai", "gemini"] = "openai"

    # ── Paths ──────────────────────────────────────────────────────────────
    working_dir: str = "./eval/rag_storage"   # LightRAG index lives here
    corpus_dir: str = "./eval/corpus"          # Drop your .txt/.md docs here
    questions_file: str = "./eval/questions.json"
    results_dir: str = "./eval/results"

    # ── Ollama settings (used when backend="ollama") ───────────────────────
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"
    embedding_model: str = "bge-m3:latest"
    embedding_dim: int = 1024

    # ── OpenAI settings (used when backend="openai") ───────────────────────
    openai_api_key: str = ""                   # or set OPENAI_API_KEY env var
    openai_llm_model: str = "gpt-4o-mini"      # cheap + capable; swap to gpt-4o for best quality
    openai_embed_model: str = "text-embedding-3-small"
    openai_embed_dim: int = 1536               # text-embedding-3-small native dim

    # ── Gemini settings (used when backend="gemini") ───────────────────────
    gemini_api_key: str = ""                   # or set GEMINI_API_KEY env var
    gemini_llm_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "models/text-embedding-004"
    gemini_embed_dim: int = 768

    # ── Judge LLM ──────────────────────────────────────────────────────────
    # Defaults to the same model as the backend LLM.
    # Override with EVAL_JUDGE_MODEL to use a smarter/cheaper model for judging.
    judge_model: str = ""   # empty = auto-select based on backend

    # ── Chunking (experiment variables) ────────────────────────────────────
    chunk_token_size: int = 1200          # tokens per chunk (default LightRAG value)
    chunk_overlap_token_size: int = 100   # overlap between consecutive chunks

    # ── Query settings ─────────────────────────────────────────────────────
    modes: list = field(
        default_factory=lambda: ["naive", "local", "global", "hybrid", "mix"]
    )
    top_k: int = 60
    chunk_top_k: int = 20          # 10 was too low — caused all modes to converge
    max_total_tokens: int = 30000  # matches CLAUDE.md recommendation
    enable_rerank: bool = False     # set True to unlock mix-mode advantage

    # ── Pipeline behaviour ─────────────────────────────────────────────────
    skip_ingestion: bool = False   # True = reuse existing index
    force_reindex: bool = False    # True = wipe + rebuild from corpus
    max_parallel_insert: int = 2

    def experiment_tag(self) -> str:
        """Short slug that uniquely identifies an (embed_model, chunk_size) combo.
        Used as the working_dir suffix so each experiment has its own isolated index."""
        if self.backend == "openai":
            model_slug = self.openai_embed_model.replace("/", "-").replace(".", "-")
        elif self.backend == "gemini":
            model_slug = self.gemini_embed_model.split("/")[-1]
        else:
            model_slug = self.embedding_model.replace(":", "-").replace("/", "-")
        return f"{model_slug}__chunk{self.chunk_token_size}"

    def effective_judge_model(self) -> str:
        if self.judge_model and self.judge_model.strip():
            return self.judge_model
        default = {
            "openai": self.openai_llm_model,
            "gemini": self.gemini_llm_model,
            "ollama": self.llm_model,
        }[self.backend]
        import warnings
        warnings.warn(
            f"judge_model is not set — falling back to '{default}'. "
            "This model is too lenient and produces flat scores. "
            "Set judge_model='gpt-4o' (or equivalent) explicitly.",
            stacklevel=2,
        )
        return default

    def effective_api_key(self) -> str:
        """Return the API key for the active backend."""
        if self.backend == "openai":
            return self.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if self.backend == "gemini":
            return self.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        return ""


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
        "openai_api_key":       "OPENAI_API_KEY",
        "openai_llm_model":     "EVAL_OPENAI_LLM_MODEL",
        "openai_embed_model":   "EVAL_OPENAI_EMBED_MODEL",
        "openai_embed_dim":     "EVAL_OPENAI_EMBED_DIM",
        "gemini_api_key":       "GEMINI_API_KEY",
        "gemini_llm_model":     "EVAL_GEMINI_LLM_MODEL",
        "gemini_embed_model":   "EVAL_GEMINI_EMBED_MODEL",
        "gemini_embed_dim":     "EVAL_GEMINI_EMBED_DIM",
        "judge_model":          "EVAL_JUDGE_MODEL",
        "top_k":                "EVAL_TOP_K",
        "chunk_top_k":          "EVAL_CHUNK_TOP_K",
        "max_total_tokens":     "EVAL_MAX_TOTAL_TOKENS",
        "enable_rerank":        "EVAL_ENABLE_RERANK",
        "skip_ingestion":       "EVAL_SKIP_INGESTION",
        "force_reindex":        "EVAL_FORCE_REINDEX",
        "max_parallel_insert":  "EVAL_MAX_PARALLEL_INSERT",
    }

    int_fields = {
        "embedding_dim", "openai_embed_dim", "gemini_embed_dim",
        "top_k", "chunk_top_k", "max_total_tokens", "max_parallel_insert",
    }
    bool_fields = {"skip_ingestion", "force_reindex", "enable_rerank"}

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
