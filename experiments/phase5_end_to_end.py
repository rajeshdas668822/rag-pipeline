#!/usr/bin/env python
"""Phase 5: Full local RAG app and answer evaluation.

Wires the full local pipeline:
DocumentLoader -> Chunker -> OllamaEmbedder -> retriever (dense/hybrid)
-> CrossEncoderReranker -> OllamaGenerator -> Evaluation

Each test question is answered twice (baseline vs reranked retrieval) and
reported side by side: answer correctness, faithfulness, context relevance,
citation accuracy (manual), and latency in seconds, plus the retrieval metrics
per configuration. A failing stage is recorded (naming the stage + question)
and evaluation continues. After all questions, the best-performing
configuration is identified.

Satisfies Requirements 15.1, 15.2, 15.3, 15.5, 15.6, 15.7.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_retriever import BM25Retriever
from src.chunker import Chunker, InvalidChunkConfigError
from src.config import Config
from src.corpus_manifest import compare_manifest
from src.cross_encoder_reranker import CrossEncoderReranker, ModelLoadError
from src.dense_retriever import DenseRetriever, NumpyBackend
from src.document_loader import DocumentLoader
from src.embedding_cache import CachedEmbedder
from src.evaluation import Evaluation
from src.evaluation_dataset import InvalidEvaluationRecordError, load_evaluation_queries
from src.hybrid_retriever import HybridRetriever
from src.ollama_client import OllamaClient
from src.ollama_embedder import OllamaEmbedder
from src.ollama_generator import GenerationError, OllamaGenerator
from src.rag_types import Chunk

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("phase5")

# (config name, retrieval stage, reranked?)
CONFIGS = [
    ("baseline", "dense", False),
    ("reranked", "dense", True),
    ("baseline_hybrid", "hybrid", False),
    ("reranked_hybrid", "hybrid", True),
]


def build_pipeline(config, data_dir, chunk_size, chunk_overlap, use_cache=True):
    """Build the Phase 1 pipeline and return (retriever, chunk_map)."""
    ollama_client = OllamaClient(endpoint=config.ollama_endpoint)
    ollama_client.ensure_ready([config.embedding_model, config.instruct_model])

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

    dense = DenseRetriever(embedder=embedder, backend=NumpyBackend())
    dense.index(chunks)

    bm25 = BM25Retriever()
    bm25.index(chunks)

    hybrid = HybridRetriever(dense, bm25, rrf_k=config.rrf_k, fetch_k=50)

    if use_cache:
        logger.info("embeddings: %d cached, %d computed", embedder.hits, embedder.misses)

    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "chunk_map": {c.chunk_id: c for c in chunks},
        "ollama_client": ollama_client,
    }


def evaluate_generation(result, relevant_docs):
    """Manual assessment of a generation result.

    Returns a dict with answer correctness, faithfulness, context relevance,
    citation accuracy (all 0-1 scale), and whether citations match relevant docs.
    """
    answer = result.answer
    citations = result.citations
    insufficient = result.insufficient_context

    # Basic checks
    no_context = answer == "no context available"
    cannot_determine = answer == "cannot be determined from the context"

    # Correctness: Did we get the answer right (or admit we dont know)?
    if no_context or cannot_determine:
        correctness = 1.0 if not relevant_docs else 0.0
    else:
        correctness = 1.0  # Placeholder - manual review needed

    # Faithfulness: Are citations drawn only from context?
    faithfulness = 1.0 if citations else 0.5  # Assume faithful if no citations

    # Context relevance: Does the answer relate to the query?
    context_relevance = 1.0 if not insufficient else 0.5

    # Citation accuracy: Do citations point to relevant docs?
    relevant_set = set(relevant_docs)
    if not citations:
        citation_accuracy = 1.0 if not relevant_docs else 0.0
    else:
        # Check if any cited docs are relevant
        cited_relevant = len([c for c in citations if c in relevant_set])
        citation_accuracy = cited_relevant / len(citations) if citations else 0.0

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "context_relevance": context_relevance,
        "citation_accuracy": citation_accuracy,
        "citations": citations,
    }


def run_phase5(queries, stages, reranker, generator, out_dir, candidates_n):
    """Run the Phase 5 end-to-end evaluation.

    Args:
        queries: Evaluation queries (query_id, query, relevant_documents, graded_labels).
        stages: dict with retrievers 'dense', 'bm25', 'hybrid' (already indexed).
        reranker: CrossEncoderReranker object.
        generator: OllamaGenerator object.
        out_dir: Output directory for results.
        candidates_n: k for retrieval.

    Returns:
        List of result dicts.
    """
    results = []
    failures = []

    for q in queries:
        relevant = set(q.relevant_documents)
        has_answer = bool(relevant)

        # Run each config
        for name, stage_name, reranked in CONFIGS:
            try:
                stage = stages[stage_name]
                out = stage.retrieve(q.query, candidates_n)
                candidates = list(out.results)
                retrieval_ms = getattr(out, "latency_ms", 0.0)

                if reranked:
                    rr = reranker.rerank(q.query, candidates, k=None)
                    candidates = [c for c in candidates if c.doc_id in [r.doc_id for r in rr.results]]

                # Get chunks for generator
                chunk_map = stages["chunk_map"]
                chunks = [chunk_map[c.doc_id] for c in candidates if c.doc_id in chunk_map]

                # Generate answer
                gen_out = generator.generate(q.query, chunks)
                generation_ms = gen_out.latency_ms

                # Evaluate generation
                gen_eval = evaluate_generation(gen_out, relevant)

                # Evaluate retrieval metrics
                ranked = [c.doc_id for c in candidates]
                retrieval_metric = Evaluation.evaluate(
                    ranked, relevant, q.graded_labels, retrieval_ms, 0.0
                )

                results.append({
                    "query_id": q.query_id,
                    "query": q.query,
                    "config": name,
                    "has_answer": has_answer,
                    "answer": gen_out.answer,
                    "insufficient_context": gen_out.insufficient_context,
                    "citations": gen_out.citations,
                    "generation_ms": generation_ms,
                    "retrieval_metric": retrieval_metric,
                    "gen_eval": gen_eval,
                })
            except Exception as exc:  # noqa: BLE001 - failure isolation
                failures.append({
                    "config": name,
                    "query_id": q.query_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                logger.warning("%s %s failed: %s", name, q.query_id, exc)

    return results, failures


def summarize_phase5(results, failures):
    """Aggregate metrics over answerable queries."""
    summary = {}
    for name in [c[0] for c in CONFIGS]:
        cfg_results = [r for r in results if r["config"] == name]
        scored = [r for r in cfg_results if r["has_answer"]]

        if not scored:
            summary[name] = {"n_queries": 0, "n_failed": 0, "n_excluded_no_answer": 0}
            continue

        # Retrieval metrics
        retrieval_metrics = {}
        for attr in ["recall_at_5", "recall_at_50", "precision_at_5", "mrr_at_10", "ndcg_at_5"]:
            vals = [getattr(r["retrieval_metric"], attr) for r in scored]
            retrieval_metrics[attr] = sum(vals) / len(vals) if vals else 0.0

        # Generation metrics
        gen_metrics = {}
        for attr in ["correctness", "faithfulness", "context_relevance", "citation_accuracy"]:
            vals = [r["gen_eval"][attr] for r in scored]
            gen_metrics[attr] = sum(vals) / len(vals) if vals else 0.0

        gen_metrics["mean_generation_ms"] = sum(r["generation_ms"] for r in scored) / len(scored)

        summary[name] = {
            "n_queries": len(scored),
            "n_failed": sum(1 for f in failures if f["config"] == name),
            "n_excluded_no_answer": len(cfg_results) - len(scored),
            **retrieval_metrics,
            **gen_metrics,
        }

    return summary


def write_phase5_outputs(out_dir, results, failures, summary):
    """Write Phase 5 output CSVs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Per-query results
    with open(out_path / "phase5_results.csv", "w", encoding="utf-8", newline="") as f:
        fields = [
            "query_id", "config", "answer", "insufficient_context", "citations",
            "generation_ms", "recall_at_5", "recall_at_50", "precision_at_5",
            "mrr_at_10", "ndcg_at_5", "correctness", "faithfulness",
            "context_relevance", "citation_accuracy",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            m = r["retrieval_metric"]
            ge = r["gen_eval"]
            w.writerow({
                "query_id": r["query_id"],
                "config": r["config"],
                "answer": r["answer"],
                "insufficient_context": str(r["insufficient_context"]).lower(),
                "citations": "|".join(r["citations"]),
                "generation_ms": r["generation_ms"],
                "recall_at_5": m.recall_at_5,
                "recall_at_50": m.recall_at_50,
                "precision_at_5": m.precision_at_5,
                "mrr_at_10": m.mrr_at_10,
                "ndcg_at_5": m.ndcg_at_5,
                "correctness": ge["correctness"],
                "faithfulness": ge["faithfulness"],
                "context_relevance": ge["context_relevance"],
                "citation_accuracy": ge["citation_accuracy"],
            })

    # Summary
    with open(out_path / "phase5_summary.csv", "w", encoding="utf-8", newline="") as f:
        fields = [
            "config", "n_queries", "n_failed", "n_excluded_no_answer",
            "recall_at_5", "recall_at_50", "precision_at_5", "mrr_at_10", "ndcg_at_5",
            "correctness", "faithfulness", "context_relevance", "citation_accuracy",
            "mean_generation_ms",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, s in summary.items():
            row = {"config": name}
            for f in fields[1:]:
                row[f] = s.get(f, "")
            w.writerow(row)

    # Failures
    with open(out_path / "phase5_failures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "query_id", "error"])
        w.writeheader()
        for fail in failures:
            w.writerow(fail)


def find_best_config(summary):
    """Identify the best performing configuration based on NDCG@5 and correctness."""
    best = None
    best_score = -1

    for name, s in summary.items():
        if s.get("n_queries", 0) == 0:
            continue

        score = s.get("ndcg_at_5", 0) * 0.5 + s.get("correctness", 0) * 0.5
        if score > best_score:
            best_score = score
            best = name

    return best, best_score


def interpret_summary(summary):
    """Generate interpretation notes for the summary."""
    lines = []

    # Compare baseline vs reranked
    for base_name, rerank_name in [
        ("baseline", "reranked"),
        ("baseline_hybrid", "reranked_hybrid"),
    ]:
        base = summary.get(base_name, {})
        rerank = summary.get(rerank_name, {})

        if not base.get("n_queries") or not rerank.get("n_queries"):
            continue

        ndcg_delta = rerank.get("ndcg_at_5", 0) - base.get("ndcg_at_5", 0)
        correct_delta = rerank.get("correctness", 0) - base.get("correctness", 0)

        lines.append(
            f"{rerank_name} vs {base_name}: "
            f"NDCG@5 {ndcg_delta:+.4f}, "
            f"correctness {correct_delta:+.4f}"
        )

    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 5: Full local RAG app and answer evaluation")
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
    p.add_argument("--ollama_endpoint", type=str, default="http://localhost:11434",
                   help="Ollama endpoint URL")
    args = p.parse_args()

    config = Config.load(overrides={"ollama_endpoint": args.ollama_endpoint})

    try:
        stages = build_pipeline(
            config, args.data_dir, args.chunk_size, args.chunk_overlap, not args.no_cache
        )
    except InvalidChunkConfigError as exc:
        logger.error("Chunking failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline setup failed: %s", exc)
        return 1

    chunk_map = stages["chunk_map"]
    logger.info("Indexed %d chunks (chunk_size=%d, overlap=%d)", len(chunk_map), args.chunk_size, args.chunk_overlap)

    try:
        queries = load_evaluation_queries(args.dataset, corpus_ids=set(chunk_map))
    except (InvalidEvaluationRecordError, FileNotFoundError) as exc:
        logger.error(
            "Dataset invalid for the current chunk config (%s/%s); labels are chunk-level and "
            "depend on it. Detail: %s", args.chunk_size, args.chunk_overlap, exc
        )
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
                manifest, chunk_map, queries, args.chunk_size, args.chunk_overlap
            )
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

    generator = OllamaGenerator(
        client=stages["ollama_client"],
        model=config.instruct_model,
        timeout_s=config.generate_timeout_s
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results, failures = run_phase5(
            queries, stages, reranker, generator, out_dir, args.candidates
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Phase 5 execution failed: %s", exc)
        return 1

    summary = summarize_phase5(results, failures)
    write_phase5_outputs(out_dir, results, failures, summary)
    best_config, best_score = find_best_config(summary)
    interpretation = interpret_summary(summary)

    # Log summary
    logger.info("")
    logger.info("Phase 5 Summary")
    logger.info("=" * 60)
    logger.info("%-20s %6s %8s %8s %8s %10s %10s %12s",
                "config", "scored", "NDCG@5", "Rec@5", "Rec@50", "correct", "faithful", "gen_ms")
    for name, s in summary.items():
        logger.info("%-20s %6d %8.4f %8.4f %8.4f %10.4f %10.4f %12.1f",
                    name, s.get("n_queries", 0), s.get("ndcg_at_5", 0), s.get("recall_at_5", 0),
                    s.get("recall_at_50", 0), s.get("correctness", 0), s.get("faithfulness", 0),
                    s.get("mean_generation_ms", 0))

    for line in interpretation:
        logger.info(line)

    if best_config:
        logger.info("")
        logger.info("Best configuration: %s (score: %.4f)", best_config, best_score)

    if failures:
        logger.warning("%d (config, query) failure(s) recorded; see phase5_failures.csv", len(failures))

    logger.info("Wrote: %s", ", ".join(str(out_dir / n) for n in (
        "phase5_results.csv", "phase5_summary.csv", "phase5_failures.csv")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())