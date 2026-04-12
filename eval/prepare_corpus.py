"""
Convert the Celery documentation JSON corpus into individual .txt files
suitable for LightRAG ingestion.

Usage:
    python -m eval.prepare_corpus --input /path/to/celery.json
    python -m eval.prepare_corpus --input /path/to/celery.json --output ./eval/corpus

Each JSON entry is expected to have:
    id, source, title, url, section, content

History/changelog entries (section == "history") are excluded by default
because they add noise without useful multihop reasoning material.
Pass --include-history to keep them.
"""

import argparse
import json
import re
from pathlib import Path


# Sections to skip — changelogs contain lots of version strings, ticket
# references, and contributor names that confuse the entity extractor.
_NOISY_SECTIONS = {"history"}


def _sanitize_filename(text: str) -> str:
    """Turn arbitrary text into a safe filename slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


def _format_doc(entry: dict) -> str:
    """Format a single corpus entry as plain text."""
    parts = []
    if entry.get("title"):
        parts.append(f"Title: {entry['title']}")
    if entry.get("url"):
        parts.append(f"URL: {entry['url']}")
    if entry.get("section"):
        parts.append(f"Section: {entry['section']}")
    parts.append("")  # blank line before body
    parts.append(entry.get("content", "").strip())
    return "\n".join(parts)


def convert(input_path: str, output_dir: str, include_history: bool = False) -> int:
    """
    Read celery.json, write one .txt per entry into output_dir.
    Returns the number of files written.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = json.loads(src.read_text(encoding="utf-8"))

    # Support both a bare list and {"docs": [...]} / {"entries": [...]} wrappers
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        # Try common wrapper keys
        for key in ("docs", "entries", "documents", "data", "results"):
            if key in data and isinstance(data[key], list):
                entries = data[key]
                break
        else:
            # Fall back to the first list value found
            entries = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        raise ValueError(f"Unexpected JSON structure in {input_path}")

    if not entries:
        raise ValueError(f"No entries found in {input_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    section_counts: dict[str, int] = {}

    for entry in entries:
        section = (entry.get("section") or "").lower()

        if not include_history and section in _NOISY_SECTIONS:
            skipped += 1
            continue

        content = (entry.get("content") or "").strip()
        if not content:
            skipped += 1
            continue

        # Build filename from id or title
        doc_id = entry.get("id") or entry.get("source") or ""
        if doc_id:
            slug = _sanitize_filename(str(doc_id))
        else:
            slug = _sanitize_filename(entry.get("title", f"doc_{written:04d}"))

        filename = f"{slug}.txt"
        dest = out / filename

        # If there's a collision, append a counter
        counter = 1
        while dest.exists():
            dest = out / f"{slug}_{counter}.txt"
            counter += 1

        dest.write_text(_format_doc(entry), encoding="utf-8")
        written += 1
        section_counts[section] = section_counts.get(section, 0) + 1

    print(f"Written : {written} files → {out}")
    print(f"Skipped : {skipped} (history/empty)")
    print("By section:")
    for sec, count in sorted(section_counts.items()):
        print(f"  {sec or '(none)':<30} {count:>4} docs")

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Celery JSON corpus to .txt files")
    p.add_argument("--input", "-i", required=True, help="Path to celery.json")
    p.add_argument(
        "--output", "-o",
        default="./eval/corpus",
        help="Output directory for .txt files (default: ./eval/corpus)",
    )
    p.add_argument(
        "--include-history",
        action="store_true",
        help="Include changelog/history entries (excluded by default)",
    )
    args = p.parse_args()
    convert(args.input, args.output, include_history=args.include_history)


if __name__ == "__main__":
    main()
