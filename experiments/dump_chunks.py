#!/usr/bin/env python
"""Dump every chunk of the corpus to JSONL for label review (no Ollama needed).

Usage:
    python experiments/dump_chunks.py [--data_dir data] [--chunk_size 512]
        [--chunk_overlap 64] [--out data/chunks_dump.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pypdf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import Chunker
from src.config import Config
from src.document_loader import DocumentLoader


def build_manifest(chunk_texts: dict, chunk_size: int, chunk_overlap: int, sources: list) -> dict:
    """Build the corpus manifest (config, count, pypdf version, per-chunk sha256)."""
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunk_texts),
        "pypdf_version": pypdf.__version__,
        "source_files": sources,
        "chunk_text_sha256": {
            cid: hashlib.sha256(text.encode("utf-8")).hexdigest() for cid, text in chunk_texts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump corpus chunks to JSONL")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--out", type=str, default="data/chunks_dump.jsonl")
    parser.add_argument("--manifest", type=str, default="data/chunks_manifest.json")
    args = parser.parse_args()

    documents = DocumentLoader(Config.load()).load_directory(args.data_dir)
    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    count = 0
    chunk_texts: dict[str, str] = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in documents:
            for chunk in chunker.chunk(doc):
                f.write(json.dumps({"chunk_id": chunk.chunk_id, "text": chunk.text}, ensure_ascii=False) + "\n")
                chunk_texts[chunk.chunk_id] = chunk.text
                count += 1
    sources = sorted({Path(d.source_path).name for d in documents})
    manifest = build_manifest(chunk_texts, args.chunk_size, args.chunk_overlap, sources)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Documents: {[d.doc_id for d in documents]}")
    print(f"Wrote {count} chunks to {args.out}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
