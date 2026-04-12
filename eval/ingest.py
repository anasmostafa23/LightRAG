"""
Document ingestor.
Reads every .txt and .md file in corpus_dir and inserts them into LightRAG.
Supports two backends: "ollama" (local) and "gemini" (Google API, free tier).
"""

import shutil
from functools import partial
from pathlib import Path

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, logger

from .config import EvalConfig


# ── Backend: Ollama ───────────────────────────────────────────────────────────

def _build_rag_ollama(cfg: EvalConfig) -> LightRAG:
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed

    return LightRAG(
        working_dir=cfg.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=cfg.llm_model,
        llm_model_kwargs={
            "host": cfg.ollama_host,
            "options": {"num_ctx": 32768},
            "timeout": 600,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=cfg.embedding_model,
                host=cfg.ollama_host,
            ),
        ),
        max_parallel_insert=cfg.max_parallel_insert,
    )


# ── Backend: Gemini ───────────────────────────────────────────────────────────

def _build_rag_gemini(cfg: EvalConfig) -> LightRAG:
    import numpy as np
    from lightrag.llm.gemini import gemini_model_complete, gemini_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    api_key = cfg.gemini_api_key
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY environment variable "
            "or cfg.gemini_api_key. Get a free key at https://aistudio.google.com/app/apikey"
        )

    llm_model = cfg.gemini_llm_model
    embed_model = cfg.gemini_embed_model

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await gemini_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            model_name=llm_model,
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=cfg.gemini_embed_dim,
        max_token_size=2048,
    )
    async def embed_func(texts: list[str]) -> np.ndarray:
        return await gemini_embed.func(texts, api_key=api_key, model=embed_model)

    return LightRAG(
        working_dir=cfg.working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        max_parallel_insert=cfg.max_parallel_insert,
    )


# ── Routing ───────────────────────────────────────────────────────────────────

def _build_rag(cfg: EvalConfig) -> LightRAG:
    if cfg.backend == "gemini":
        return _build_rag_gemini(cfg)
    return _build_rag_ollama(cfg)


# ── Corpus loading ────────────────────────────────────────────────────────────

def _collect_docs(corpus_dir: str) -> list[tuple[str, str]]:
    """Return list of (file_path, text) for every .txt / .md in corpus_dir."""
    corpus = Path(corpus_dir)
    if not corpus.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Create it and drop your document files inside."
        )

    docs = []
    for ext in ("*.txt", "*.md"):
        for p in sorted(corpus.glob(ext)):
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append((str(p), text))

    if not docs:
        raise ValueError(
            f"No .txt or .md files found in {corpus_dir}.\n"
            "Add your documents before running the pipeline."
        )
    return docs


# ── Public API ────────────────────────────────────────────────────────────────

async def ingest(cfg: EvalConfig) -> LightRAG:
    """
    Initialise LightRAG, optionally wipe old index, insert all corpus docs.
    Returns the initialised LightRAG instance (ready for querying).
    """
    working_dir = Path(cfg.working_dir)

    if cfg.force_reindex and working_dir.exists():
        logger.info(f"force_reindex=True — wiping {working_dir}")
        shutil.rmtree(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Backend: {cfg.backend.upper()}")
    rag = _build_rag(cfg)
    await rag.initialize_storages()

    if cfg.skip_ingestion:
        logger.info("skip_ingestion=True — using existing index, no documents inserted.")
        return rag

    docs = _collect_docs(cfg.corpus_dir)
    logger.info(f"Ingesting {len(docs)} document(s) from {cfg.corpus_dir} …")

    texts = [text for _, text in docs]
    file_paths = [path for path, _ in docs]

    await rag.ainsert(texts, file_paths=file_paths)

    logger.info("Ingestion complete.")
    return rag
