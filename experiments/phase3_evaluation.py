#!/usr/bin/env python
"""Phase 3: scientific Baseline vs Reranked evaluation.

For each labelled query ONE candidate set is retrieved. Baseline = that set in
retriever order. Reranked = the cross-encoder reordering of the SAME full set
(k=None, nothing truncated before scoring). Both are scored with Evaluation.

Writes to --out_dir: baseline_results.csv, reranked_results.csv,
evaluation_summary.csv.

Usage:
    python experiments/phase3_evaluation.py --repeat_check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import Chunker, InvalidChunkConfigError
from src.config import Config
from src.corpus_manifest import compare_manifest
from src.cross_encoder_reranker import CrossEncoderReranker, ModelLoadError, RerankerError
from src.dense_retriever import DenseRetriever, NumpyBackend
from src.document_loader import DocumentLoader
from src.evaluation import Evaluation, MetricResult
from src.evaluation_dataset import InvalidEvaluationRecordError, load_evaluation_queries
from src.ollama_client import OllamaClient
from src.ollama_embedder import OllamaEmbedder
from src.embedding_cache import CachedEmbedder
from src.rag_types import Chunk

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("phase3")

METRICS = [
    ("NDCG@5", "ndcg_at_5"),
    ("Recall@5", "recall_at_5"),
    ("Recall@50", "recall_at_50"),
    ("Precision@5", "precision_at_5"),
    ("MRR@10", "mrr_at_10"),
]
EPS = 1e-9

INTERPRETATION_NOTE = (
    "NDCG@5 up with Recall@50 unchanged = reranker reordered the same candidate set. "
    "Recall@5 up = relevant docs moved into the top 5 (not discovered by the reranker)."
)


def build_pipeline(config: Config, data_dir: str, chunk_size: int, chunk_overlap: int, use_cache: bool = True):
    """Build the Phase 1 pipeline and return (retriever, chunk_map)."""
    ollama_client = OllamaClient(endpoint=config.ollama_endpoint)
    ollama_client.ensure_ready([config.embedding_model])

    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    documents = DocumentLoader(config).load_directory(data_dir)

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    embedder = OllamaEmbedder(
        client=ollama_client, model=config.embedding_model, timeout_s=config.embed_timeout_s
    )
    if use_cache:
        embedder = CachedEmbedder(embedder)
    retriever = DenseRetriever(embedder=embedder, backend=NumpyBackend())
    retriever.index(chunks)
    if use_cache:
        logger.info("embeddings: %d cached, %d computed", embedder.hits, embedder.misses)
    return retriever, {c.chunk_id: c for c in chunks}


def evaluate_all(queries, retriever, reranker, candidates_n):
    """Return (rows) list of dicts with baseline/reranked ranked lists + metrics."""
    rows = []
    for q in queries:
        relevant = set(q.relevant_documents)
        out = retriever.retrieve(q.query, candidates_n)
        candidates = list(out.results)
        retrieval_ms = getattr(out, "latency_ms", 0.0)
        base_ranked = [c.doc_id for c in candidates]

        rr = reranker.rerank(q.query, candidates, k=None)
        rerank_ranked = [r.doc_id for r in rr.results]

        if set(base_ranked) != set(rerank_ranked) or len(base_ranked) != len(rerank_ranked):
            logger.error("%s: candidate sets differ between configs; no comparison reported.", q.query_id)
            continue

        base = Evaluation.evaluate(base_ranked, relevant, q.graded_labels, retrieval_ms, 0.0)
        rer = Evaluation.evaluate(rerank_ranked, relevant, q.graded_labels, retrieval_ms, rr.latency_ms)
        rows.append({
            "query_id": q.query_id, "query": q.query, "has_answer": bool(relevant),
            "base_ranked": base_ranked, "rerank_ranked": rerank_ranked,
            "base": base, "rerank": rer,
        })
    return rows


def write_results(path: Path, rows, key_ranked: str, key_metric: str) -> None:
    fields = ["query_id", "query", "document_ids", "recall_at_5", "recall_at_50",
              "precision_at_5", "mrr_at_10", "ndcg_at_5", "retrieval_latency_ms",
              "reranking_latency_ms"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m: MetricResult = r[key_metric]
            w.writerow({
                "query_id": r["query_id"], "query": r["query"],
                "document_ids": "|".join(r[key_ranked][:10]),
                "recall_at_5": m.recall_at_5, "recall_at_50": m.recall_at_50,
                "precision_at_5": m.precision_at_5, "mrr_at_10": m.mrr_at_10,
                "ndcg_at_5": m.ndcg_at_5,
                "retrieval_latency_ms": m.retrieval_latency_ms,
                "reranking_latency_ms": m.reranking_latency_ms,
            })


def summarize(rows):
    """Average metrics over queries that have at least one relevant doc."""
    scored = [r for r in rows if r["has_answer"]]
    excluded = len(rows) - len(scored)
    summary = []
    for name, attr in METRICS:
        b = sum(getattr(r["base"], attr) for r in scored) / len(scored) if scored else 0.0
        t = sum(getattr(r["rerank"], attr) for r in scored) / len(scored) if scored else 0.0
        summary.append((name, b, t))
    return summary, len(scored), excluded


def interpret(name, b, t):
    delta = t - b
    if abs(delta) <= EPS:
        return f"{name} unchanged."
    direction = "up" if delta > 0 else "down"
    if name == "Recall@50":
        return "Recall@50 differs; unexpected since candidate sets are identical."
    if name == "NDCG@5":
        return (f"NDCG@5 {direction} {abs(delta):.4f} with Recall@50 unchanged = "
                "reranker reordered the same candidate set.")
    if name == "Recall@5":
        return (f"Recall@5 {direction} {abs(delta):.4f}: relevant docs "
                f"{'moved into' if delta > 0 else 'moved out of'} the top 5 "
                "(not discovered by the reranker).")
    return f"{name} {direction} {abs(delta):.4f} (ranking order change within the same candidates)."


def write_summary(path: Path, summary) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "reranked", "changed", "interpretation"])
        for name, b, t in summary:
            changed = abs(t - b) > EPS
            w.writerow([name, f"{b:.6f}", f"{t:.6f}", str(changed).lower(), interpret(name, b, t)])


def metric_fingerprint(rows):
    return [(r["query_id"], r["base_ranked"], r["rerank_ranked"],
             tuple((k, v) for k, v in asdict(r["base"]).items() if "latency" not in k),
             tuple((k, v) for k, v in asdict(r["rerank"]).items() if "latency" not in k))
            for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3: baseline vs reranked evaluation")
    p.add_argument("--candidates", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--dataset", type=str, default="data/evaluation_queries.jsonl")
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--manifest", type=str, default="data/chunks_manifest.json")
    p.add_argument("--skip_manifest_check", action="store_true",
                   help="Do not abort when the corpus differs from the one the labels were drafted on")
    p.add_argument("--no_cache", action="store_true", help="Bypass the on-disk embedding cache")
    p.add_argument("--repeat_check", action="store_true",
                   help="Run the evaluation twice and assert identical metrics (Req 11.6)")
    args = p.parse_args()

    config = Config.load()
    try:
        retriever, chunk_map = build_pipeline(config, args.data_dir, args.chunk_size, args.chunk_overlap, not args.no_cache)
    except InvalidChunkConfigError as exc:
        logger.error("Chunking failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline setup failed: %s", exc)
        return 1
    logger.info("Indexed %d chunks (chunk_size=%d, overlap=%d)", len(chunk_map), args.chunk_size, args.chunk_overlap)

    try:
        queries = load_evaluation_queries(args.dataset, corpus_ids=set(chunk_map))
    except (InvalidEvaluationRecordError, FileNotFoundError) as exc:
        logger.error(
            "Dataset invalid for the current chunk config (%s/%s); labels are chunk-level and "
            "depend on it. Detail: %s", args.chunk_size, args.chunk_overlap, exc)
        return 1

    if not args.skip_manifest_check:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            logger.warning("Corpus manifest %s not found; cannot verify the corpus matches the labels.", manifest_path)
        else:
            import pypdf
            with open(manifest_path, encoding="utf-8") as mf:
                manifest = json.load(mf)
            problems, mismatched = compare_manifest(
                manifest, chunk_map, queries, args.chunk_size, args.chunk_overlap)
            if problems:
                logger.error("Corpus drift: current chunks do not match the corpus the labels were drafted on.")
                for problem in problems:
                    logger.error("  - %s", problem)
                logger.error("  manifest: %s chunks, pypdf %s; current: %d chunks, pypdf %s",
                             manifest.get("chunk_count"), manifest.get("pypdf_version"),
                             len(chunk_map), pypdf.__version__)
                if mismatched:
                    logger.error("  first mismatching ids: %s", mismatched[:5])
                logger.error("Fix: use the same pypdf version and PDF as the manifest, or rerun "
                             "experiments/dump_chunks.py and relabel. (Override: --skip_manifest_check)")
                return 1

    try:
        reranker = CrossEncoderReranker(model_name=config.cross_encoder_model)
    except ModelLoadError as exc:
        logger.error("Cross-encoder model could not be loaded: %s", exc)
        return 1
    reranker.index(chunk_map)

    try:
        rows = evaluate_all(queries, retriever, reranker, args.candidates)
    except RerankerError as exc:
        logger.error("Reranking failed: %s", exc)
        return 1

    for r in rows:
        logger.info("%s  ndcg@5 %.3f -> %.3f", r["query_id"], r["base"].ndcg_at_5, r["rerank"].ndcg_at_5)

    summary, n_scored, n_excluded = summarize(rows)
    # Candidate sets are identical, so Recall@50 must match exactly.
    for name, b, t in summary:
        if name == "Recall@50":
            assert abs(b - t) <= EPS, f"Recall@50 differs ({b} vs {t}) despite identical candidate sets"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results(out_dir / "baseline_results.csv", rows, "base_ranked", "base")
    write_results(out_dir / "reranked_results.csv", rows, "rerank_ranked", "rerank")
    write_summary(out_dir / "evaluation_summary.csv", summary)

    logger.info("")
    logger.info("Summary over %d queries with answers (%d no-answer queries excluded from averages)",
                n_scored, n_excluded)
    logger.info("%-12s %10s %10s  %s", "metric", "baseline", "reranked", "changed")
    for name, b, t in summary:
        logger.info("%-12s %10.4f %10.4f  %s", name, b, t, str(abs(t - b) > EPS).lower())
    logger.info("Recall@50 identical between configs (same candidate set): verified.")
    for name, b, t in summary:
        logger.info(interpret(name, b, t))
    logger.info("Wrote: %s, %s, %s", out_dir / "baseline_results.csv",
                out_dir / "reranked_results.csv", out_dir / "evaluation_summary.csv")

    if args.repeat_check:
        first = metric_fingerprint(rows)
        rows2 = evaluate_all(queries, retriever, reranker, args.candidates)
        second = metric_fingerprint(rows2)
        assert first == second, "Repeat run produced different metrics (non-deterministic)"
        logger.info("Repeat check: second in-process run produced identical metrics and rankings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
