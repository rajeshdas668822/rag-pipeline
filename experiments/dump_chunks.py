#!/usr/bin/env python
"""Dump every chunk of the corpus to JSONL for label review (no Ollama needed).

Usage:
    python experiments/dump_chunks.py [--data_dir data] [--chunk_size 512]
        [--chunk_overlap 64] [--out data/chunks_dump.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import Chunker
from src.config import Config
from src.document_loader import DocumentLoader


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump corpus chunks to JSONL")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--out", type=str, default="data/chunks_dump.jsonl")
    args = parser.parse_args()

    documents = DocumentLoader(Config.load()).load_directory(args.data_dir)
    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in documents:
            for chunk in chunker.chunk(doc):
                f.write(json.dumps({"chunk_id": chunk.chunk_id, "text": chunk.text}, ensure_ascii=False) + "\n")
                count += 1
    print(f"Documents: {[d.doc_id for d in documents]}")
    print(f"Wrote {count} chunks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
