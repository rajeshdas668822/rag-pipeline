#!/usr/bin/env python
"""Phase 2: Cross-encoder reranking experiment.

This script reuses the Phase 1 retrieval wiring to build a candidate set, then
applies the local cross-encoder reranker and records how each document moved:

  Config -> OllamaClient -> DocumentLoader -> Chunker -> OllamaEmbedder
         -> DenseRetriever (candidate set) -> CrossEncoderReranker (top 5)

It writes the Phase 2 CSV report (via src.phase2_report) with one row per
candidate per query and prints the written path.

Experiment modes (--mode):
    initial   (a) initial retrieval only, no reranking
    20        (b) retrieve 20 and rerank 20
    50        (c) retrieve 50 and rerank 50
    100       (d) retrieve 100 and rerank 100
    reworded  (e) rerank the candidate set of a reworded query (--reworded)
    irrelevant(f) rerank a candidate set with an inserted irrelevant document

Usage:
    python experiments/phase2_reranking.py -q "What is RAG?" --mode 20
    python experiments/phase2_reranking.py -q "What is RAG?" --mode reworded \
        --reworded "Explain retrieval augmented generation"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import Chunker, InvalidChunkConfigError
from src.config import Config
from src.cross_encoder_reranker import CrossEncoderReranker, ModelLoadError, RerankerError
from src.dense_retriever import DenseRetriever, NumpyBackend
from src.document_loader import DocumentLoader
from src.ollama_client import OllamaClient
from src.ollama_embedder import OllamaEmbedder
from src.phase2_report import write_phase2_report
from src.rag_types import Chunk, RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("phase2")

MODE_TOP_K = {
    "initial": 10,
    "20": 20,
    "50": 50,
    "100": 100,
    "reworded": 20,
    "irrelevant": 20,
}

# A synthetic, clearly off-topic document inserted for experiment (f).
IRRELEVANT_CHUNK = Chunk(
    chunk_id="irrelevant#inserted",
    doc_id="irrelevant",
    text=(
        "The recipe for a classic Victoria sponge cake calls for equal parts "
        "butter, sugar, flour, and eggs, baked at 180 degrees Celsius."
    ),
    seq=0,
)


def validate_query(query: str) -> None:
    if not query:
        raise ValueError("Query must be non-empty.")
    if len(query) > 2048:
        raise ValueError("Query must not exceed 2048 characters.")


def build_pipeline(config: Config, data_dir: str, chunk_size: int, chunk_overlap: int):
    """Build the Phase 1 pipeline and return (retriever, chunk_map)."""
    ollama_client = OllamaClient(endpoint=config.ollama_endpoint)
    ollama_client.ensure_ready([config.embedding_model])

    loader = DocumentLoader(config)
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    documents = loader.load_directory(data_dir)

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    embedder = OllamaEmbedder(
        client=ollama_client,
        model=config.embedding_model,
        timeout_s=config.embed_timeout_s,
    )
    retriever = DenseRetriever(embedder=embedder, backend=NumpyBackend())
    retriever.index(chunks)

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    return retriever, chunk_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: Cross-encoder reranking experiment"
    )
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "--mode",
        type=str,
        default="20",
        choices=sorted(MODE_TOP_K.keys()),
        help="Experiment mode (default: 20)",
    )
    parser.add_argument(
        "--reworded",
        type=str,
        default=None,
        help="Reworded query text (used when --mode reworded)",
    )
    parser.add_argument(
        "--rerank_k", type=int, default=5, help="Top-k reranked docs (default: 5)"
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--query_id", type=str, default="q1", help="query_id for the CSV report"
    )
    parser.add_argument(
        "--relevant",
        type=str,
        nargs="*",
        default=[],
        help="Manually labelled relevant doc_ids (chunk ids)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: data/phase2_report_<mode>.csv)",
    )
    args = parser.parse_args()

    try:
        validate_query(args.query)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    retrieve_query = args.query
    if args.mode == "reworded":
        if not args.reworded:
            logger.error("--mode reworded requires --reworded <text>")
            return 1
        try:
            validate_query(args.reworded)
        except ValueError as exc:
            logger.error("Configuration error (reworded): %s", exc)
            return 1
        retrieve_query = args.reworded

    config = Config.load()
    logger.info("Ollama endpoint: %s", config.ollama_endpoint)

    # Build the Phase 1 pipeline / candidate set.
    try:
        retriever, chunk_map = build_pipeline(
            config, args.data_dir, args.chunk_size, args.chunk_overlap
        )
    except InvalidChunkConfigError as exc:
        logger.error("Chunking failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline setup failed: %s", exc)
        return 1

    top_k = MODE_TOP_K[args.mode]
    retrieval_output = retriever.retrieve(retrieve_query, top_k)
    candidates = list(retrieval_output.results)

    # Experiment (f): insert an irrelevant document into the candidate set.
    if args.mode == "irrelevant":
        chunk_map[IRRELEVANT_CHUNK.chunk_id] = IRRELEVANT_CHUNK
        candidates.append(
            RetrievalResult(
                doc_id=IRRELEVANT_CHUNK.chunk_id,
                score=0.0,
                rank=len(candidates) + 1,
            )
        )

    logger.info('Query: "%s"', args.query)
    logger.info("Mode: %s | candidate set size: %d", args.mode, len(candidates))

    # Requirement 8.8: empty / unavailable candidate set -> descriptive error, no report.
    if not candidates:
        logger.error(
            "Candidate set is unavailable (empty) for query %r; no report written.",
            args.query,
        )
        return 1

    relevant_ids = set(args.relevant)
    # Requirement 8.7: record relevant docs absent from the candidate set.
    candidate_ids = {c.doc_id for c in candidates}
    absent_relevant = relevant_ids - candidate_ids
    if absent_relevant:
        logger.info(
            "Relevant docs absent from candidate set (and so from reranked results): %s",
            sorted(absent_relevant),
        )

    out_path = args.out or f"data/phase2_report_{args.mode}.csv"

    if args.mode == "initial":
        # (a) Initial retrieval only, no reranking. Report candidates with no
        # rerank scores/positions (left blank by the writer).
        from src.cross_encoder_reranker import RerankOutput

        rerank_output = RerankOutput(results=[], latency_ms=0.0)
        logger.info("Initial-only mode: reranker not applied.")
    else:
        # Load the cross-encoder and rerank.
        try:
            reranker = CrossEncoderReranker(model_name=config.cross_encoder_model)
        except ModelLoadError as exc:
            logger.error("Cross-encoder model could not be loaded: %s", exc)
            return 1

        reranker.index(chunk_map)
        try:
            rerank_output = reranker.rerank(args.query, candidates, k=args.rerank_k)
        except RerankerError as exc:
            logger.error("Reranking failed: %s", exc)
            return 1

        logger.info(
            "Reranked top %d (latency %.2f ms):",
            len(rerank_output.results),
            rerank_output.latency_ms,
        )
        for r in rerank_output.results:
            logger.info(
                "  pos %d <- initial_rank %d | %s | rerank_score=%.4f initial_score=%.4f",
                r.reranked_position,
                r.initial_rank,
                r.doc_id,
                r.rerank_score,
                r.initial_score,
            )

    written = write_phase2_report(
        output_path=out_path,
        query_id=args.query_id,
        candidates=candidates,
        rerank_output=rerank_output,
        relevant_doc_ids=relevant_ids,
    )
    logger.info("Phase 2 CSV report written to: %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
