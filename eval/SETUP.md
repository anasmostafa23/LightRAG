# Evaluation Pipeline — Setup & Usage

## What's in this directory

```
eval/
├── config.py       — all knobs (models, paths, modes, etc.)
├── ingest.py       — reads corpus/ docs → builds LightRAG index
├── query.py        — runs each question across naive/local/global/hybrid/mix
├── judge.py        — LLM-as-judge scores each answer (faithfulness, relevance, completeness, coherence)
├── report.py       — writes CSV + JSON, prints summary table
├── pipeline.py     — orchestrates everything, CLI entry point
├── questions.json  — YOUR multihop Q&A pairs (edit this)
└── corpus/         — YOUR documents go here (.txt or .md)
```

Results land in `eval/results/results_YYYYMMDD_HHMMSS.{csv,json}`.

---

## Step 1 — Install dependencies (one-time)

```bash
cd /home/mx/workspace/github.com/anasmostafa23/LightRAG

# Install LightRAG + Ollama bindings + all core deps
pip install -e ".[offline-llm]"
```

---

## Step 2 — Start Ollama and pull models

```bash
# In a separate terminal (keep it running)
ollama serve

# Pull the LLM used for indexing + querying
ollama pull qwen2.5:14b          # or any model with ≥ 32k context

# Pull the embedding model (MUST stay the same across runs)
ollama pull bge-m3:latest        # 1024-dim, works well for RAG

# The judge model can be the same or different
# (already pulled above if you're using qwen2.5:14b)
```

> **Note**: If you use a different model, update `llm_model`, `embedding_model`,
> and `embedding_dim` in `eval/config.py` before indexing.

---

## Step 3 — Add your documents

Drop `.txt` or `.md` files in `eval/corpus/`:

```
eval/corpus/
├── doc_about_topic_A.txt
├── doc_about_topic_B.txt
└── linking_A_and_B.md
```

For meaningful multihop evaluation:
- Each document should cover a distinct aspect of your domain.
- Good multihop questions require *at least two* documents to answer.

---

## Step 4 — Fill in your multihop questions

Edit `eval/questions.json`. The format:

```json
{
  "questions": [
    {
      "id": "q1",
      "hop_count": 2,
      "question": "How does X influence Y, given that Z is involved?",
      "ground_truth": "Complete, correct answer that covers both hops.",
      "metadata": { "topic": "optional", "difficulty": "medium" }
    },
    {
      "id": "q2",
      "hop_count": 3,
      "question": "Three-hop question chaining facts from three different documents.",
      "ground_truth": "Full answer covering all three reasoning steps.",
      "metadata": { "topic": "optional", "difficulty": "hard" }
    }
  ]
}
```

**Required fields**: `id`, `question`, `ground_truth`
**Optional fields**: `hop_count` (informational, stored in results), `metadata`

---

## Step 5 — Run the pipeline

```bash
cd /home/mx/workspace/github.com/anasmostafa23/LightRAG

# Full run: ingest corpus → query all modes → score → report
python -m eval.pipeline

# Reuse an existing index (skip re-indexing)
python -m eval.pipeline --skip-ingestion

# Wipe and rebuild the index from scratch
python -m eval.pipeline --force-reindex

# Only run specific modes
python -m eval.pipeline --modes local,hybrid,mix --skip-ingestion

# Collect raw answers without judging (faster, no scoring)
python -m eval.pipeline --no-judge --skip-ingestion

# Custom questions file
python -m eval.pipeline --questions path/to/my_questions.json
```

---

## Configuration overrides (environment variables)

All `eval/config.py` values can be overridden without editing the file:

| Variable              | Default                   | What it controls               |
|-----------------------|---------------------------|--------------------------------|
| `EVAL_LLM_MODEL`      | `qwen2.5:14b`             | LightRAG indexing/query model  |
| `EVAL_EMBEDDING_MODEL`| `bge-m3:latest`           | Embedding model                |
| `EVAL_EMBEDDING_DIM`  | `1024`                    | Embedding vector dimension     |
| `EVAL_JUDGE_MODEL`    | `qwen2.5:14b`             | Judge LLM for scoring          |
| `EVAL_OLLAMA_HOST`    | `http://localhost:11434`  | Ollama server URL              |
| `EVAL_MODES`          | all five modes            | Comma-separated mode list      |
| `EVAL_TOP_K`          | `60`                      | KG entities retrieved          |
| `EVAL_CHUNK_TOP_K`    | `10`                      | Text chunks retrieved          |
| `EVAL_SKIP_INGESTION` | `false`                   | Reuse existing index           |
| `EVAL_FORCE_REINDEX`  | `false`                   | Wipe + rebuild index           |

---

## Scoring dimensions (1–5 scale)

| Metric            | What it measures                                      |
|-------------------|-------------------------------------------------------|
| `faithfulness`    | Are claims grounded in facts? (no hallucinations)     |
| `answer_relevance`| Does the answer directly address the question?        |
| `completeness`    | Are all reasoning hops covered in the answer?         |
| `coherence`       | Is the reasoning clear and logically structured?      |
| `overall`         | Mean of the four above                                |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'httpx'`**  
→ Run `pip install -e ".[offline-llm]"` first.

**`ConnectionError` to Ollama**  
→ Make sure `ollama serve` is running in another terminal.

**`No .txt or .md files found in eval/corpus/`**  
→ Add your documents to `eval/corpus/` before running.

**Empty answers from LightRAG**  
→ The index might be empty. Run without `--skip-ingestion` at least once.

**Judge returns all zeros**  
→ The judge model may not be following the JSON output format.  
   Try a larger model: `EVAL_JUDGE_MODEL=qwen2.5:32b python -m eval.pipeline --skip-ingestion`
