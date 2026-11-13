#!/usr/bin/env python
"""Phase 4: retrieval comparison (Part A) and reranker comparison (Part B).

Part A compares five configs over identical queries and labels:
dense, bm25, dense+cross_encoder, hybrid, hybrid+cross_encoder.
Each retrieval stage is queried ONCE per query (k=--candidates); configs sharing a
stage use the identical candidate set. Rerank configs reorder the full set (k=None).

Part B compares the cross-encoder with the (experimental) LLM reranker on the SAME
truncated candidate set, and measures latency, model evaluations, memory and
top-5 consistency across repeated runs.

Writes to --out_dir: phase4_retrieval_comparison.csv, phase4_per_query.csv,
phase4_reranker_comparison.csv, phase4_failures.csv.

Usage:
    python experiments/phase4_retrieval_comparison.py --skip_llm
    python experiments/phase4_retrieval_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
import tracemalloc
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
from src.ollama_llm_reranker import OllamaLLMReranker
from src.rag_types import Chunk

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("phase4")

# (config name, retrieval stage, reranked?)
CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("dense+cross_encoder", "dense", True),
    ("hybrid", "hybrid", False),
    ("hybrid+cross_encoder", "hybrid", True),
]
METRICS = [
    ("Recall@5", "recall_at_5"),
    ("Recall@50", "recall_at_50"),
    ("Precision@5", "precision_at_5"),
    ("MRR@10", "mrr_at_10"),
    ("NDCG@5", "ndcg_at_5"),
]
EPS = 1e-9
MEMORY_SCOPE = "this process only; excludes the Ollama server process"


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------- Part A

def run_retrieval_comparison(queries, stages, reranker, candidates_n):
    """Run the five-config comparison.

    Args:
        queries: Evaluation queries (query_id, query, relevant_documents, graded_labels).
        stages: dict with retrievers 'dense', 'bm25', 'hybrid' (already indexed).
        reranker: object with rerank(query, candidates, k=None) (cross-encoder).
        candidates_n: k used for every retrieval stage.

    Returns:
        (rows, failures). rows: one dict per successful (config, query) with
        query_id, config, has_answer, ranked, metric. failures: dicts with
        config, query_id, error.
    """
    rows, failures = [], []
    for q in queries:
        relevant = set(q.relevant_documents)
        has_answer = bool(relevant)
        retrieved = {}  # stage -> (candidates, retrieval_ms) or Exception
        for stage in ("dense", "bm25", "hybrid"):
            try:
                out = stages[stage].retrieve(q.query, candidates_n)
                retrieved[stage] = (list(out.results), getattr(out, "latency_ms", 0.0))
            except Exception as exc:  # noqa: BLE001 - failure isolation
                retrieved[stage] = exc

        query_rows = {}
        for name, stage, reranked in CONFIGS:
            try:
                got = retrieved[stage]
                if isinstance(got, Exception):
                    raise got
                candidates, retrieval_ms = got
                rerank_ms = 0.0
                ranked = [c.doc_id for c in candidates]
                if reranked:
                    rr = reranker.rerank(q.query, candidates, k=None)
                    ranked = [r.doc_id for r in rr.results]
                    rerank_ms = rr.latency_ms
                    if sorted(ranked) != sorted(c.doc_id for c in candidates):
                        raise RuntimeError("reranker output differs from its candidate set")
                metric = Evaluation.evaluate(ranked, relevant, q.graded_labels, retrieval_ms, rerank_ms)
            except Exception as exc:  # noqa: BLE001 - failure isolation (Req 12.6)
                failures.append({"config": name, "query_id": q.query_id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            row = {"query_id": q.query_id, "config": name, "has_answer": has_answer,
                   "ranked": ranked, "metric": metric}
            rows.append(row)
            query_rows[name] = row

        # Same candidate set => identical Recall@50 for base vs reranked.
        for base, rer in (("dense", "dense+cross_encoder"), ("hybrid", "hybrid+cross_encoder")):
            if base in query_rows and rer in query_rows:
                a, b = query_rows[base]["metric"].recall_at_50, query_rows[rer]["metric"].recall_at_50
                assert abs(a - b) <= EPS, (
                    f"{q.query_id}: Recall@50 differs for {base} vs {rer} ({a} vs {b}) "
                    "despite identical candidate sets")
        logger.info("%s%s  ndcg@5 %s", q.query_id, "" if has_answer else " (no-answer)",
                    "  ".join(f"{n}={query_rows[n]['metric'].ndcg_at_5:.3f}" if n in query_rows else f"{n}=FAILED"
                              for n, _, _ in CONFIGS))
    return rows, failures


def summarize_retrieval(rows, failures, n_queries):
    """Aggregate per config over answerable queries; no-answer queries are excluded."""
    summary = []
    for name, _, reranked in CONFIGS:
        cfg_rows = [r for r in rows if r["config"] == name]
        scored = [r for r in cfg_rows if r["has_answer"]]
        entry = {
            "config": name,
            "n_queries_scored": len(scored),
            "n_excluded_no_answer": len(cfg_rows) - len(scored),
            "n_failed": sum(1 for f in failures if f["config"] == name),
            "n_queries_total": n_queries,
        }
        for label, attr in METRICS:
            entry[attr] = _mean(getattr(r["metric"], attr) for r in scored)
        entry["mean_retrieval_ms"] = _mean(r["metric"].retrieval_latency_ms for r in scored)
        entry["mean_rerank_ms"] = _mean(r["metric"].reranking_latency_ms for r in scored)
        summary.append(entry)
    by_name = {s["config"]: s for s in summary}
    for base, rer in (("dense", "dense+cross_encoder"), ("hybrid", "hybrid+cross_encoder")):
        a, b = by_name[base], by_name[rer]
        if a["n_queries_scored"] == b["n_queries_scored"]:
            assert abs(a["recall_at_50"] - b["recall_at_50"]) <= EPS, (
                f"Recall@50 differs for {base} vs {rer} despite identical candidate sets")
    return summary


def interpret_retrieval(summary):
    by_name = {s["config"]: s for s in summary}
    lines = []
    live = [s for s in summary if s["n_queries_scored"] > 0]
    if not live:
        return ["No config scored any answerable query."]
    for label, attr in METRICS:
        best = max(live, key=lambda s: s[attr])
        ties = [s["config"] for s in live if abs(s[attr] - best[attr]) <= EPS]
        lines.append(f"Best {label}: {' = '.join(ties)} ({best[attr]:.4f})")
    d, h = by_name["dense"], by_name["hybrid"]
    delta = h["recall_at_50"] - d["recall_at_50"]
    if abs(delta) <= EPS:
        lines.append("Hybrid Recall@50 equals dense: fusion did not surface additional relevant docs.")
    elif delta > 0:
        lines.append(f"Hybrid Recall@50 is {delta:.4f} above dense: retrieval found more relevant docs "
                     "(a retrieval gain, not a reranking reorder).")
    else:
        lines.append(f"Hybrid Recall@50 is {-delta:.4f} below dense: fusion lost relevant docs from the candidates.")
    for base, rer in (("dense", "dense+cross_encoder"), ("hybrid", "hybrid+cross_encoder")):
        a, b = by_name[base], by_name[rer]
        lines.append(f"{rer} vs {base}: NDCG@5 {b['ndcg_at_5'] - a['ndcg_at_5']:+.4f}, "
                     f"Recall@5 {b['recall_at_5'] - a['recall_at_5']:+.4f}, Recall@50 unchanged "
                     "(same candidates; any gain is reordering).")
    if len({s["n_queries_scored"] for s in live}) > 1:
        lines.append("Note: configs scored different numbers of queries because of failures; "
                     "averages are not over identical query sets.")
    return lines


def write_retrieval_outputs(out_dir, rows, failures, summary):
    with open(out_dir / "phase4_retrieval_comparison.csv", "w", encoding="utf-8", newline="") as f:
        fields = (["config", "n_queries_scored", "n_failed", "n_excluded_no_answer"]
                  + [a for _, a in METRICS] + ["mean_retrieval_ms", "mean_rerank_ms"])
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in s.items()})
    with open(out_dir / "phase4_per_query.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "config", "has_answer"] + [a for _, a in METRICS]
                   + ["retrieval_ms", "rerank_ms", "top10_ids"])
        for r in rows:
            m = r["metric"]
            w.writerow([r["query_id"], r["config"], str(r["has_answer"]).lower()]
                       + [getattr(m, a) for _, a in METRICS]
                       + [m.retrieval_latency_ms, m.reranking_latency_ms, "|".join(r["ranked"][:10])])
    write_failures(out_dir, failures)


def write_failures(out_dir, failures):
    with open(out_dir / "phase4_failures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "query_id", "error"])
        w.writeheader()
        for fail in failures:
            w.writerow(fail)


# ---------------------------------------------------------------- Part B

def _rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _evaluations(reranker, name, pairs_scored):
    return reranker.calls if name == "llm" else pairs_scored


def run_reranker_comparison(queries, base_retriever, rerankers, candidates_n, llm_candidates,
                            llm_queries, consistency_runs, base_name):
    """Compare rerankers on the same truncated candidate set.

    Args:
        queries: Evaluation queries; only answerable ones are used.
        base_retriever: Indexed retriever for the rerank base stage.
        rerankers: dict name -> reranker (e.g. {'cross_encoder': ce, 'llm': llm}).
            An object with reset_stats()/calls/parse_failures is treated as an LLM-style
            counter; others count pairs scored (= candidates per call).
        candidates_n: k for the base retrieval before truncation.
        llm_candidates: Candidates kept per query for both rerankers.
        llm_queries: Number of answerable queries to use (None = all).
        consistency_runs: Repeats of the identical rerank (>= 3).
        base_name: 'hybrid' or 'dense' (for labelling).

    Returns:
        (result_rows, failures).
    """
    if consistency_runs < 3:
        raise ValueError("consistency_runs must be at least 3")
    answerable = [q for q in queries if q.relevant_documents]
    if llm_queries is not None:
        answerable = answerable[:llm_queries]
    failures = []

    cand_sets = {}
    for q in answerable:
        try:
            out = base_retriever.retrieve(q.query, candidates_n)
            cand_sets[q.query_id] = list(out.results)[:llm_candidates]
        except Exception as exc:  # noqa: BLE001
            failures.append({"config": f"rerank_base:{base_name}", "query_id": q.query_id,
                             "error": f"{type(exc).__name__}: {exc}"})
    usable = [q for q in answerable if q.query_id in cand_sets]

    calls_est = len(usable) * llm_candidates * consistency_runs
    logger.info("Part B: %d queries x %d candidates x %d runs = ~%d model evaluations per reranker",
                len(usable), llm_candidates, consistency_runs, calls_est)
    logger.info("  LLM runtime estimate: %d calls; at 1-3 s/call on CPU that is roughly %.0f-%.0f minutes "
                "(measure the first query to refine)", calls_est, calls_est / 60, calls_est * 3 / 60)

    # per reranker -> query_id -> {"runs": [ranked lists], "ms": [..], "rr_first": results}
    data = {name: {} for name in rerankers}
    stats = {name: {"peak_mb": 0.0, "rss_delta": None, "evals": 0, "parse_failures": 0,
                    "pairs": 0} for name in rerankers}
    for name, rr in rerankers.items():
        if hasattr(rr, "reset_stats"):
            rr.reset_stats()

    for idx, q in enumerate(usable, start=1):
        cands = cand_sets[q.query_id]
        logger.info("[%d/%d] %s: reranking %d candidates", idx, len(usable), q.query_id, len(cands))
        for name, rr in rerankers.items():
            runs, times = [], []
            try:
                for run in range(consistency_runs):
                    # Run 1 is traced for memory; later runs are untraced so latency
                    # excludes tracemalloc overhead (and first-call model warm-up).
                    if run == 0:
                        rss_before = _rss_mb()
                        tracemalloc.start()
                    try:
                        out = rr.rerank(q.query, cands, k=None)
                    finally:
                        if run == 0:
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            stats[name]["peak_mb"] = max(stats[name]["peak_mb"], peak / (1024 * 1024))
                            rss_after = _rss_mb()
                            if rss_before is not None and rss_after is not None:
                                prev = stats[name]["rss_delta"] or 0.0
                                stats[name]["rss_delta"] = max(prev, rss_after - rss_before)
                    stats[name]["pairs"] += len(cands)
                    ranked = [r.doc_id for r in out.results]
                    if sorted(ranked) != sorted(c.doc_id for c in cands):
                        raise RuntimeError("reranker output differs from its candidate set")
                    runs.append((ranked, [r for r in out.results]))
                    if run > 0:
                        times.append(out.latency_ms)
            except Exception as exc:  # noqa: BLE001 - failure isolation
                failures.append({"config": f"reranker:{name}", "query_id": q.query_id,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            data[name][q.query_id] = {"runs": [r[0] for r in runs], "ms": times,
                                      "results": runs[0][1]}

    common = [q for q in usable if all(q.query_id in data[n] for n in rerankers)]
    if len(common) < len(usable):
        logger.warning("Part B: %d/%d queries scored by every reranker; comparison uses those only",
                       len(common), len(usable))

    result_rows = []
    pre_recalls = {}
    for name, rr in rerankers.items():
        ndcgs, mrrs, recalls, ms_all, ident, jacc = [], [], [], [], [], []
        for q in common:
            d = data[name][q.query_id]
            relevant = set(q.relevant_documents)
            first = d["runs"][0]
            m = Evaluation.evaluate(first, relevant, q.graded_labels, 0.0, 0.0)
            ndcgs.append(m.ndcg_at_5)
            mrrs.append(m.mrr_at_10)
            recalls.append(m.recall_at_50)  # reranked set == candidate set => pre-rerank Recall@50
            ms_all.extend(d["ms"])
            tops = [r[:5] for r in d["runs"]]
            ident.append(all(t == tops[0] for t in tops))
            jacc.extend(_jaccard(a, b) for a, b in itertools.combinations(tops, 2))
        pre_recalls[name] = _mean(recalls)
        evals = rr.calls if hasattr(rr, "calls") else stats[name]["pairs"]
        fails = getattr(rr, "parse_failures", None)
        is_llm = hasattr(rr, "calls")
        notes = ("experimental; pointwise 0-3 LLM scores tie often (ties keep retriever order)"
                 if is_llm else "deterministic local model")
        notes += f"; memory is {MEMORY_SCOPE}; latency from runs 2..{consistency_runs}"
        if stats[name]["rss_delta"] is None:
            notes += "; RSS delta skipped (psutil not installed)"
        result_rows.append({
            "reranker": name, "rerank_base": base_name, "n_queries": len(common),
            "n_candidates": llm_candidates,
            "ndcg_at_5": _mean(ndcgs), "mrr_at_10": _mean(mrrs),
            "pre_rerank_recall_at_50": pre_recalls[name],
            "mean_rerank_ms": _mean(ms_all),
            "model_evaluations": evals,
            "parse_failures": fails if fails is not None else 0,
            "tracemalloc_peak_mb": stats[name]["peak_mb"],
            "rss_delta_mb": stats[name]["rss_delta"] if stats[name]["rss_delta"] is not None else "",
            "consistency_runs": consistency_runs,
            "top5_identical_fraction": _mean(1.0 if x else 0.0 for x in ident),
            "mean_pairwise_jaccard": _mean(jacc),
            "notes": notes,
        })
    vals = list(pre_recalls.values())
    assert all(abs(v - vals[0]) <= EPS for v in vals), (
        f"pre-rerank Recall@50 differs between rerankers {pre_recalls} despite identical candidates")
    return result_rows, failures


RERANKER_FIELDS = [
    "reranker", "rerank_base", "n_queries", "n_candidates", "ndcg_at_5", "mrr_at_10",
    "pre_rerank_recall_at_50", "mean_rerank_ms", "model_evaluations", "parse_failures",
    "tracemalloc_peak_mb", "rss_delta_mb", "consistency_runs", "top5_identical_fraction",
    "mean_pairwise_jaccard", "notes",
]


def write_reranker_csv(out_dir, result_rows):
    with open(out_dir / "phase4_reranker_comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RERANKER_FIELDS)
        w.writeheader()
        for r in result_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def interpret_rerankers(result_rows):
    if len(result_rows) < 2:
        return []
    by = {r["reranker"]: r for r in result_rows}
    ce, llm = by.get("cross_encoder"), by.get("llm")
    if not ce or not llm:
        return []
    d = llm["ndcg_at_5"] - ce["ndcg_at_5"]
    if abs(d) <= EPS:
        verdict = "LLM and cross-encoder tie on NDCG@5"
    elif d > 0:
        verdict = f"LLM scored {d:.4f} higher NDCG@5 than the cross-encoder on this sample"
    else:
        verdict = f"LLM scored {-d:.4f} lower NDCG@5 than the cross-encoder on this sample"
    lines = [
        f"{verdict} (n={ce['n_queries']} queries; small sample, treat the LLM reranker as experimental).",
        f"Cost: mean rerank latency cross-encoder {ce['mean_rerank_ms']:.1f} ms vs LLM "
        f"{llm['mean_rerank_ms']:.1f} ms per query; LLM parse failures: {llm['parse_failures']}.",
        f"Consistency over {llm['consistency_runs']} runs: top-5 identical for "
        f"{ce['top5_identical_fraction']:.2%} (cross-encoder) vs {llm['top5_identical_fraction']:.2%} (LLM) "
        f"of queries; mean Jaccard {ce['mean_pairwise_jaccard']:.3f} vs {llm['mean_pairwise_jaccard']:.3f}.",
        f"Memory ({MEMORY_SCOPE}) is not comparable for the LLM, which runs inside Ollama.",
    ]
    return lines


# ---------------------------------------------------------------- setup / main

def build_stages(config, ollama_client, chunks, use_cache, candidates_n):
    embedder = OllamaEmbedder(client=ollama_client, model=config.embedding_model,
                              timeout_s=config.embed_timeout_s)
    if use_cache:
        embedder = CachedEmbedder(embedder)
    dense = DenseRetriever(embedder=embedder, backend=NumpyBackend())
    dense.index(chunks)  # embedded ONCE
    if use_cache:
        logger.info("embeddings: %d cached, %d computed", embedder.hits, embedder.misses)
    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = HybridRetriever(dense, bm25, rrf_k=config.rrf_k, fetch_k=candidates_n)
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}


def check_manifest(args, chunk_map, queries):
    """Return True when OK (or skipped); False on drift (errors already logged)."""
    if args.skip_manifest_check:
        return True
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.warning("Corpus manifest %s not found; cannot verify the corpus matches the labels.", manifest_path)
        return True
    import pypdf
    with open(manifest_path, encoding="utf-8") as mf:
        manifest = json.load(mf)
    problems, mismatched = compare_manifest(manifest, chunk_map, queries, args.chunk_size, args.chunk_overlap)
    if not problems:
        return True
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
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: retrieval and reranker comparison")
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
    p.add_argument("--skip_llm", action="store_true", help="Skip Part B (cross-encoder vs LLM reranker)")
    p.add_argument("--rerank_base", choices=["hybrid", "dense"], default="hybrid",
                   help="Retrieval stage whose candidates feed Part B")
    p.add_argument("--llm_candidates", type=int, default=20,
                   help="Top candidates per query given to both rerankers in Part B")
    p.add_argument("--llm_queries", type=int, default=None,
                   help="Use only the first N answerable queries in Part B (default: all)")
    p.add_argument("--consistency_runs", type=int, default=3,
                   help="Repeated reranks for top-5 consistency (minimum 3)")
    args = p.parse_args()
    if args.consistency_runs < 3:
        p.error("--consistency_runs must be at least 3")
    if args.llm_candidates < 1 or args.candidates < 1:
        p.error("--candidates and --llm_candidates must be >= 1")

    config = Config.load()
    try:
        client = OllamaClient(endpoint=config.ollama_endpoint)
        required = [config.embedding_model] + ([] if args.skip_llm else [config.instruct_model])
        client.ensure_ready(required)
        if not Path(args.data_dir).exists():
            raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")
        documents = DocumentLoader(config).load_directory(args.data_dir)
        chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(chunker.chunk(doc))
        stages = build_stages(config, client, chunks, not args.no_cache, args.candidates)
    except InvalidChunkConfigError as exc:
        logger.error("Chunking failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline setup failed: %s", exc)
        return 1
    chunk_map = {c.chunk_id: c for c in chunks}
    logger.info("Indexed %d chunks (chunk_size=%d, overlap=%d)", len(chunk_map), args.chunk_size, args.chunk_overlap)

    try:
        queries = load_evaluation_queries(args.dataset, corpus_ids=set(chunk_map))
    except (InvalidEvaluationRecordError, FileNotFoundError) as exc:
        logger.error(
            "Dataset invalid for the current chunk config (%s/%s); labels are chunk-level and "
            "depend on it. Detail: %s", args.chunk_size, args.chunk_overlap, exc)
        return 1
    if not check_manifest(args, chunk_map, queries):
        return 1

    try:
        cross_encoder = CrossEncoderReranker(model_name=config.cross_encoder_model)
    except ModelLoadError as exc:
        logger.error("Cross-encoder model could not be loaded: %s", exc)
        return 1
    cross_encoder.index(chunk_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Part A
    rows, failures = run_retrieval_comparison(queries, stages, cross_encoder, args.candidates)
    summary = summarize_retrieval(rows, failures, len(queries))
    n_excluded = sum(1 for q in queries if not q.relevant_documents)

    # ---- Part B
    reranker_rows = []
    if not args.skip_llm:
        llm = OllamaLLMReranker(client=client, model=config.instruct_model,
                                timeout_s=config.generate_timeout_s)
        llm.index(chunk_map)
        reranker_rows, b_failures = run_reranker_comparison(
            queries, stages[args.rerank_base], {"cross_encoder": cross_encoder, "llm": llm},
            args.candidates, args.llm_candidates, args.llm_queries, args.consistency_runs,
            args.rerank_base)
        failures = failures + b_failures
    else:
        logger.info("Part B skipped (--skip_llm)")

    write_retrieval_outputs(out_dir, rows, failures, summary)
    write_reranker_csv(out_dir, reranker_rows)

    logger.info("")
    logger.info("Part A: %d answerable queries scored per config (%d no-answer queries excluded from averages)",
                len(queries) - n_excluded, n_excluded)
    logger.info("%-22s %6s %6s %9s %10s %12s %8s %8s %10s %9s", "config", "scored", "failed",
                *[n for n, _ in METRICS], "retr_ms", "rerank_ms")
    for s in summary:
        logger.info("%-22s %6d %6d %9.4f %10.4f %12.4f %8.4f %8.4f %10.1f %9.1f",
                    s["config"], s["n_queries_scored"], s["n_failed"],
                    *[s[a] for _, a in METRICS], s["mean_retrieval_ms"], s["mean_rerank_ms"])
    logger.info("Recall@50 identical for dense vs dense+cross_encoder and hybrid vs hybrid+cross_encoder: verified.")
    for line in interpret_retrieval(summary):
        logger.info(line)

    if reranker_rows:
        logger.info("")
        logger.info("Part B: reranker comparison on the top %d of '%s' candidates (memory: %s)",
                    args.llm_candidates, args.rerank_base, MEMORY_SCOPE)
        logger.info("%-14s %8s %8s %10s %12s %7s %9s %10s %9s %9s %8s", "reranker", "NDCG@5", "MRR@10",
                    "preR@50", "rerank_ms", "evals", "parse_f", "tracemalloc", "rss_d_mb", "top5_same", "jaccard")
        for r in reranker_rows:
            rss = f"{r['rss_delta_mb']:.1f}" if r["rss_delta_mb"] != "" else "n/a"
            logger.info("%-14s %8.4f %8.4f %10.4f %12.1f %7d %9d %9.1fMB %9s %8.2f%% %8.3f",
                        r["reranker"], r["ndcg_at_5"], r["mrr_at_10"], r["pre_rerank_recall_at_50"],
                        r["mean_rerank_ms"], r["model_evaluations"], r["parse_failures"],
                        r["tracemalloc_peak_mb"], rss, 100 * r["top5_identical_fraction"],
                        r["mean_pairwise_jaccard"])
        logger.info("Pre-rerank Recall@50 identical for both rerankers (same candidates): verified.")
        for line in interpret_rerankers(reranker_rows):
            logger.info(line)

    if failures:
        logger.warning("%d (config, query) failure(s) recorded; see phase4_failures.csv", len(failures))
    logger.info("Wrote: %s", ", ".join(str(out_dir / n) for n in (
        "phase4_retrieval_comparison.csv", "phase4_per_query.csv",
        "phase4_reranker_comparison.csv", "phase4_failures.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
