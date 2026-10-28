# Implementation Plan: RAG Retrieval and Reranking Evaluation Lab

## Overview

This plan builds the lab in five sequential phases, each ending in a runnable, verifiable deliverable before the next begins. Setup tasks (1–3) establish the project layout, dependencies, configuration, and the Ollama runtime check. Task 4 defines the shared data types every component depends on. Phases 1–5 (tasks 5–9+) then build the retrieval engine, the cross-encoder reranker, scientific evaluation, the strategy/reranker comparison, and the full end-to-end RAG app.

The implementation language is Python (as specified throughout the design document). All reusable components live under `src/`, all runnable phase scripts under `experiments/`, and all data and generated reports under `data/`. Ollama-dependent components are tested against a mocked `OllamaClient` so the suite runs with no live model; live integration tests are optional and guarded by an environment flag.

Tasks marked with `*` are optional. Optional tasks cover unit/property/integration tests and the two phase-gated stretch features (FAISS backend, Ragas integration). The core, non-optional path builds a working lab end to end without them.

## Tasks

- [x] 1. Scaffold the project structure and declare dependencies
  - Create the `src/`, `experiments/`, and `data/` top-level directories, with `src/` holding reusable component modules and `experiments/` holding only runnable phase scripts (no reusable logic)
  - Add `src/__init__.py` and a `tests/` directory for the test suite
  - Declare dependencies in `pyproject.toml` (or `requirements.txt`): core `numpy`, `pandas`, `ollama`, `sentence-transformers`, `torch`, `rank-bm25`, `pypdf`; dev/test `hypothesis`, `pytest`; optional `faiss-cpu` and `ragas` as an extras group; explicitly exclude `langchain` and `llamaindex`
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 2. Implement the Config object and Ollama endpoint resolution
  - [x] 2.1 Implement `src/config.py` with the frozen `Config` dataclass and `Config.load(overrides)`
    - Resolve the Ollama Endpoint in precedence order: `OLLAMA_ENDPOINT` env var → explicit config value → `DEFAULT_OLLAMA_ENDPOINT` (`http://localhost:11434`)
    - Carry model names (`embeddinggemma`, `qwen2.5:3b`), chunk defaults (size 512, overlap 64), `dense_backend` ("numpy"|"faiss"), `use_ragas`, `rrf_k` (60), and the embed/generate timeouts
    - _Requirements: 17.1, 17.2, 17.3, 17.8_
  - [x]* 2.2 Write unit tests for endpoint resolution
    - Assert env var overrides the config value, config value overrides the default, and the default applies when unset
    - _Requirements: 17.1, 17.2, 17.3_

- [x] 3. Implement the OllamaClient reachability and model-availability check
  - [x] 3.1 Implement `src/ollama_client.py` with `OllamaClient`, `OllamaUnavailableError`, and `OllamaModelMissingError`
    - Provide `ping()`, `available_models()`, `embed(model, text)`, and `chat(model, prompt, timeout_s)` over the configured endpoint, connecting only to the local endpoint with no external/paid API fallback
    - Implement `ensure_ready(required_models)` to raise `OllamaUnavailableError` naming the endpoint when unreachable, or `OllamaModelMissingError` naming the endpoint and missing model when a model is absent
    - _Requirements: 1.6, 1.7, 17.4, 17.5, 17.6, 17.8_
  - [x]* 3.2 Write unit tests for `ensure_ready` with a mocked transport
    - Cover reachable + models present (no error), unreachable (endpoint-named error), and model missing (endpoint + model-named error)
    - _Requirements: 1.7, 17.5, 17.6_
  - [x] 3.3 Document native and Docker Ollama setup in the README
    - Document `ollama serve` + model pulls for native install, the `docker run`/`docker exec` equivalents, and the `OLLAMA_ENDPOINT` variable; state that native install is sufficient and Docker is not required
    - _Requirements: 17.7_

- [x] 4. Define the shared data types
  - Implement `src/types.py` with the frozen dataclasses `Document`, `Chunk`, `RetrievalResult`, and `RerankResult` exactly as in the design (including `RerankResult.parse_failed` defaulting to `False`)
  - These types are the contract that makes retrievers and rerankers interchangeable
  - _Requirements: 1.1, 1.5, 7.4, 7.5_

- [x] 5. Phase 1 — Build the retrieval engine from scratch
  - [x] 5.1 Implement the document loader with multi-format extraction
    - Implement `src/document_loader.py` with `DocumentLoader`, `UnsupportedFileTypeError`, `ExtractionError`, and `DuplicateDocumentIdError`
    - Detect type by extension for `.pdf` (pypdf, local, no API), `.md`/`.txt` (UTF-8), and `.json`/`.jsonl` (structured records with id + non-empty text); assign `<filename>#<seq>` IDs to non-JSON sources
    - Reject unsupported extensions, PDF-with-no-text, missing/empty/zero-record files, and duplicate IDs with a descriptive error naming the offending path, leaving the knowledge base unpopulated on failure
    - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.12_
  - [ ]* 5.2 Write unit tests for the document loader
    - Cover each extension's happy path plus the error cases (unsupported extension, PDF extraction failure, empty file, duplicate id)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.12_
  - [ ] 5.3 Assemble the labelled knowledge base under `data/`
    - Create source documents covering RAG, embeddings, vector databases, Transformer architectures, and Python so that loading yields 100–500 chunks each of length 1–2000 chars, including the five required document categories (relevant, semantically-similar-but-irrelevant, keyword-overlap, different-wording, partial-answer)
    - Record manual relevance labels for test queries, each referencing document IDs that exist in the loaded knowledge base
    - _Requirements: 2.9, 2.10, 2.11_
  - [x] 5.4 Implement the chunker
    - Implement `src/chunker.py` with `Chunker` and `InvalidChunkConfigError`; produce sequential chunks of at most `chunk_size` units with exactly `chunk_overlap` shared trailing/leading units and unique chunk IDs
    - Reject `overlap >= size`, `size < 1`, or `overlap < 0` with a range-naming error and no chunks; return zero chunks for empty text
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  - [ ]* 5.5 Write unit tests for the chunker
    - Assert exact chunk boundaries and overlap on a known string, invalid-config rejection, and empty-text returning zero chunks
    - _Requirements: 3.2, 3.3, 3.5, 3.6, 3.7_
  - [x] 5.6 Implement the Ollama embedder
    - Implement `src/ollama_embedder.py` with `OllamaEmbedder` and `EmbeddingError`; request embeddings for non-empty text up to 2048 chars via the client, exposing a fixed queryable `dimension`
    - Reject empty or >2048-char input without calling the API; raise `EmbeddingError` on 30s timeout or a malformed/wrong-dimension response without returning the bad vector
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 17.4_
  - [ ]* 5.7 Write unit tests for the embedder with a mocked client
    - Cover input validation (empty / >2048), fixed-dimension output, timeout error, and malformed/wrong-dim response rejection
    - _Requirements: 4.2, 4.3, 4.4, 4.5_
  - [x] 5.8 Implement the dense retriever with the NumPy backend
    - Implement `src/dense_retriever.py` with the `VectorBackend` protocol, `NumpyBackend` (from-scratch cosine, no external similarity/ranking lib), the `Retriever` protocol, `RetrievalOutput`, and `DenseRetriever`
    - Return top-k by descending cosine (ties by ascending index); k>N returns all N; k<1 or an empty collection returns empty results without an unhandled error; zero-magnitude or dim-mismatch embeddings raise an error with no divide-by-zero; report `latency_ms`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 16.2_
  - [ ]* 5.9 Write unit tests for the dense retriever
    - Cover cosine on hand-computed vectors, tie-breaking by ascending index, k>N, and zero-magnitude / dim-mismatch errors
    - _Requirements: 5.1, 5.2, 5.4, 5.6_
  - [ ]* 5.10 Write property test — retriever count and uniqueness
    - **Property 1: A retriever never returns more than the requested count, with no duplicates**
    - **Validates: Requirements 5.3, 16.1** — over random corpora and k, assert `len(results) <= min(k, N)` and unique ids (embedder mocked)
  - [ ]* 5.11 Write property test — empty / under-K lists handled by the dense retriever
    - **Property 6: An empty candidate list is handled without error**
    - **Validates: Requirements 5.5, 7.8, 10.6** — empty collection and k>N produce empty/available-only results with no exception
  - [ ] 5.12 Build and run `experiments/phase1_retrieval.py` (Phase 1 deliverable)
    - Wire `Config` → `OllamaClient.ensure_ready(["embeddinggemma"])` → `DocumentLoader` → `Chunker` → `OllamaEmbedder` → `DenseRetriever` (NumPy) and expose configurable query, chunk size (50–2048), and chunk overlap (0..size-1)
    - Print the query, retrieved document IDs, similarity scores to ≥4 decimals, ranked contents in descending order, and retrieval time in ms; print the BM25 keyword ranking and the dense semantic ranking side by side; reject empty/>2048-char queries and out-of-range chunk config before retrieving
    - Run the script against the `data/` knowledge base to confirm it produces the ranked-results deliverable
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ]* 5.13 Write property test — reproducibility and cosine cross-check
    - **Property 8: Results are reproducible across runs**
    - **Validates: Requirements 6.6, 11.6** — identical inputs across two runs give identical rankings/scores; cosine matches an independent NumPy recomputation within 1e-6 (embedder mocked)

- [x] 6. Checkpoint — Phase 1 complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Phase 2 — Cross-encoder reranking
  - [ ] 7.1 Implement the cross-encoder reranker
    - Implement `src/cross_encoder_reranker.py` with the `Reranker` protocol, `RerankOutput`, `CrossEncoderReranker`, `RerankerError`, and `ModelLoadError`; load `cross-encoder/ms-marco-MiniLM-L6-v2` locally via Sentence Transformers (no Ollama, no paid API), raising `ModelLoadError` with no results on load failure
    - Build exactly one query-doc pair and one score per candidate; sort by descending score with stable tie order; return `min(k, len)` when k given; reject k<1 or non-int; empty candidates → empty result with no error; treat scores as raw comparison values, not probabilities; report `latency_ms`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
  - [ ]* 7.2 Write unit tests for the cross-encoder reranker with a mocked model
    - Cover ordering, top-k slicing, invalid top-k rejection, and model-load failure
    - _Requirements: 7.2, 7.4, 7.6, 7.7_
  - [ ]* 7.3 Write property test — reranker descending sort with stable ties
    - **Property 2: A reranker returns scores sorted descending, stable for ties**
    - **Validates: Requirements 7.4, 13.4** — over random candidate sets with a mocked scorer, assert non-increasing scores and stable order for equal scores
  - [ ]* 7.4 Write property test — reranked output is a duplicate-free subset
    - **Property 3: Every reranked document came from the candidate set exactly once**
    - **Validates: Requirements 7.5, 13.4** — output ids are a duplicate-free subset of input ids and `len(output) <= len(input)`
  - [ ]* 7.5 Write property test — empty candidate list handled by the reranker
    - **Property 6: An empty candidate list is handled without error**
    - **Validates: Requirements 7.8, 5.5, 10.6** — an empty candidate set yields an empty result with no exception
  - [ ] 7.6 Implement the Phase 2 CSV report writer
    - Write a per-candidate-per-query CSV with fields `query_id, document_id, initial_rank, initial_similarity_score, rerank_score, reranked_position, is_relevant`, where `is_relevant` comes from manual labels and is never derived from reranker scores; report the written file path
    - _Requirements: 8.3, 8.4, 8.6_
  - [ ] 7.7 Build and run `experiments/phase2_reranking.py` (Phase 2 deliverable)
    - Take the Phase 1 candidate set, apply `CrossEncoderReranker.rerank(query, candidates, k=5)`, return the top-5 reranked documents, and record every candidate's before/after position pairing and score pairing
    - Support experiments (a) initial-only, (b) retrieve/rerank 20, (c) 50, (d) 100, (e) reworded query, (f) inserted irrelevant document; record a relevant doc absent from the candidate set as absent from both candidate set and reranked results; on an empty/unavailable candidate set return a descriptive error and no report
    - Run the script to produce the Phase 2 CSV report
    - _Requirements: 8.1, 8.2, 8.5, 8.7, 8.8_
  - [ ]* 7.8 Write property test — relevant doc outside the candidate set is absent from rerank output
    - **Property 4: A relevant document outside the candidate set cannot appear in reranked results**
    - **Validates: Requirements 8.7, 7.5** — build candidate sets excluding a known relevant doc and assert its absence from rerank output and its "absent" recording in the Phase 2 report path

- [x] 8. Checkpoint — Phase 2 complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Phase 3 — Scientific baseline-vs-reranked evaluation
  - [x] 9.1 Implement the evaluation metrics module
    - Implement `src/evaluation.py` with `MetricResult` and `Evaluation`; compute Recall@5, Recall@50, Precision@5, MRR@10, and NDCG@5 (graded labels when available, else binary) and report retrieval and reranking latency as two distinct ms fields
    - Define Recall and MRR as 0 for queries with no relevant docs and exclude them from averages consistently; compute over available results when a list is empty or K exceeds its length, without error
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [ ]* 9.2 Write example-based metric fixtures (the independent oracle)
    - Hand-calculate Recall@5, Precision@5, MRR@10, and NDCG@5 (including a graded-relevance example and a no-relevant-docs query) and assert each computed metric within 1e-4
    - _Requirements: 10.2_
  - [ ]* 9.3 Write property test — metric bounds, monotonicity, and ideal ordering
    - **Property 5: Evaluation metrics match independently hand-calculated examples**
    - **Validates: Requirements 10.2** — assert all metrics ∈ [0, 1], Recall@K non-decreasing in K, and ideal ordering yields NDCG@5 = 1 (exact values covered by the fixtures in 9.2)
  - [ ]* 9.4 Write property test — retrieval and reranking latency measured separately
    - **Property 7: Retrieval latency and reranking latency are measured separately**
    - **Validates: Requirements 10.3** — assert both latency fields are present, numeric, non-negative, and set independently for baseline vs reranked
  - [x] 9.5 Implement loading of the labelled evaluation dataset
    - Add a loader for `data/evaluation_queries.jsonl` producing records with `query_id`, `query`, `relevant_documents`, and optional `graded_labels` (0–3); reject duplicate `query_id`, missing/empty `query`, or a `relevant_documents` id matching no corpus document, naming the offending record and field
    - Author the dataset itself: 30–50 questions with manual labels, including ≥3 different-terminology, ≥3 multi-relevant, ≥3 same-topic/different-question, and ≥3 no-answer (empty `relevant_documents`) questions, plus graded labels for affected queries
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  - [ ]* 9.6 Write unit tests for the evaluation-dataset loader
    - Cover a valid dataset plus each rejection case (duplicate query_id, empty query, unknown relevant doc id)
    - _Requirements: 9.1, 9.6_
  - [ ] 9.7 Build and run `experiments/phase3_evaluation.py` (Phase 3 deliverable)
    - For each query obtain one initial candidate set, evaluate a Baseline (candidate set unchanged) and a Reranked (same set, cross-encoder applied) configuration over identical queries/documents/labels; error for a query whose two configs do not share an identical candidate set and report no comparison for it
    - Produce `baseline_results.csv`, `reranked_results.csv`, and `evaluation_summary.csv` (NDCG@5, Recall@5, Recall@50 for both configs, whether each metric changed, and an interpretation column documenting "NDCG@5 up / Recall@50 unchanged = reordering" and "Recall@5 up = relevant docs moved into top 5"); ensure identical metric values across repeated runs
    - Run the script to produce the three CSVs
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [x] 10. Checkpoint — Phase 3 complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Phase 4 — Retrieval strategy and reranker comparison
  - [ ] 11.1 Implement the BM25 retriever
    - Implement `src/bm25_retriever.py` with `BM25Retriever` behind the shared `Retriever` interface; rank via `rank-bm25`, returning a ranked list of 1..k (k in 1..1000) and reporting `latency_ms`
    - _Requirements: 12.2_
  - [ ]* 11.2 Write unit tests for the BM25 retriever
    - Assert ranked ordering on a small keyword corpus and the top-k bound
    - _Requirements: 12.2_
  - [ ] 11.3 Implement the hybrid retriever with RRF
    - Implement `src/hybrid_retriever.py` with `HybridRetriever` behind the shared `Retriever` interface; fuse the dense and BM25 lists via RRF with k=60 (`1/(rrf_k + rank)`), never by adding raw cosine/BM25 scores; return top-k by descending fused score and report `latency_ms`
    - _Requirements: 12.3_
  - [ ]* 11.4 Write unit tests for the hybrid retriever
    - Assert RRF fusion against a hand-worked example with k=60
    - _Requirements: 12.3_
  - [ ] 11.5 Implement the Ollama LLM reranker
    - Implement `src/ollama_llm_reranker.py` with `OllamaLLMReranker` behind the shared `Reranker` interface; prompt the instruct model with the same prompt and fixed 0–3 scale per candidate, parse responses to int 0..3, and on an unparseable response assign score 0, set `parse_failed=True`, and continue without aborting; return only received candidates sorted descending
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 17.4_
  - [ ]* 11.6 Write unit tests for the LLM reranker with a mocked client
    - Cover 0–3 parsing, the unparseable→score-0 + `parse_failed=True` path, and descending ordering over received candidates
    - _Requirements: 13.2, 13.3, 13.4_
  - [ ] 11.7 Build and run `experiments/phase4_retrieval_comparison.py` (Phase 4 deliverable)
    - Compare exactly the five configurations (dense no-rerank, BM25 no-rerank, dense + cross-encoder, hybrid no-rerank, hybrid + cross-encoder) over the identical labelled question set and identical candidate set for configs sharing a retrieval stage, recording Recall@5, Recall@50, Precision@5, MRR@10, NDCG@5 each; on a failing config record the config + affected question, exclude it, and continue
    - Compare the cross-encoder vs the LLM reranker over the same candidate set and questions, recording NDCG@5, MRR@10, pre-rerank Recall@50, reranking latency (ms), peak memory, number of model evaluations, and top-5 ranking consistency across ≥3 repeated runs, treating the LLM reranker as experimental
    - Run the script to produce the comparison report
    - _Requirements: 12.1, 12.4, 12.5, 12.6, 13.5, 13.6, 13.7_
  - [ ]* 11.8 Implement the optional FAISS backend (stretch)
    - Implement `FaissBackend` in `src/dense_retriever.py` behind the same `VectorBackend` interface with identical public signatures, enabled via `Config.dense_backend="faiss"`; on init/search failure raise a "vector search backend unavailable" error preserving the index with no partial results; reject k<1 or a zero-document index
    - _Requirements: 16.1, 16.2, 16.4, 16.5_
  - [ ]* 11.9 Write property test — FAISS matches the NumPy reference (stretch)
    - **Supplementary Property: FAISS matches the NumPy reference**
    - **Validates: Requirements 16.3** — FAISS top-k ordering matches NumPy for ≥99% of positions with per-document scores within 0.0001; `@skipif` FAISS not installed

- [ ] 12. Checkpoint — Phase 4 complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Phase 5 — Full local RAG app and answer evaluation
  - [ ] 13.1 Implement the Ollama generator
    - Implement `src/ollama_generator.py` with `OllamaGenerator`, `GenerationResult`, and `GenerationError`; build a context+query prompt instructing the model to answer only from context and cite supporting document IDs, call the Ollama chat API, and return an answer with citations drawn only from supplied chunks
    - Return "no context available" without calling the API on empty chunks; return "cannot be determined from the context" when context lacks the answer; raise `GenerationError` on 60s timeout/failure with no fabricated answer
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 17.4_
  - [ ]* 13.2 Write unit tests for the generator with a mocked client
    - Cover the empty-chunk short-circuit, insufficient-context refusal, citation mapping to supplied chunks, and the timeout error
    - _Requirements: 14.2, 14.4, 14.5, 14.6_
  - [ ] 13.3 Build and run `experiments/phase5_end_to_end.py` (Phase 5 deliverable)
    - Wire the full local pipeline: `DocumentLoader` → `Chunker` → `OllamaEmbedder` → retriever (dense/hybrid) → `CrossEncoderReranker` → `OllamaGenerator` → `Evaluation`, running entirely on the local machine
    - Answer each test question twice (baseline vs reranked retrieval) and report side by side answer correctness, faithfulness, context relevance, citation accuracy, and latency in seconds, plus the retrieval metrics per configuration; on a failing stage record the stage + question and continue; after all questions identify the best-performing configuration
    - Run the script to produce the end-to-end report
    - _Requirements: 15.1, 15.2, 15.3, 15.5, 15.6, 15.7_
  - [ ]* 13.4 Integrate optional Ragas answer evaluation (stretch)
    - When `Config.use_ragas=True`, compute Ragas answer-quality metrics in addition to the manual assessment in `phase5_end_to_end.py`, leaving the manual path as the default when Ragas is disabled
    - _Requirements: 15.4_

- [ ] 14. Checkpoint — Phase 5 complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 15. Optional live-Ollama integration tests (stretch)
  - Add integration tests for the embedder, generator, and LLM reranker that exercise the real local Ollama instance, guarded by an environment flag and a live-endpoint reachability check, skipped automatically when the endpoint is unreachable (requires pulling `embeddinggemma` and the instruct model first)
  - _Requirements: 4.1, 14.1, 13.1, 17.5_

## Notes

- Tasks marked with `*` are optional: all unit/property/integration tests and the two phase-gated stretch features (FAISS backend in 11.8/11.9, Ragas in 13.4, live integration in 15). The core non-optional path builds a working lab end to end.
- Each phase ends in a checkpoint and a runnable deliverable, so the five-phase progression is verifiable at every milestone.
- Property-based tests (Hypothesis, ≥100 iterations each) cover the eight correctness properties; each is annotated with its property number and the requirement clause it validates. Property 5's exact values come from the hand-calculated fixtures in 9.2.
- Ollama-dependent components (embedder, generator, LLM reranker) are tested against a mocked `OllamaClient` so the suite runs with no live model; live tests are optional (task 15).
- Each task references specific requirement sub-criteria for traceability.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "4", "3.3"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "5.1", "5.4", "9.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.5", "5.6", "7.1", "9.2", "9.5", "11.1", "11.3"] },
    { "id": 5, "tasks": ["5.7", "5.8", "7.2", "7.3", "7.4", "7.5", "9.3", "9.4", "9.6", "11.2", "11.4", "11.5"] },
    { "id": 6, "tasks": ["5.9", "5.10", "5.11", "7.6", "11.6", "11.8"] },
    { "id": 7, "tasks": ["5.12", "7.7", "13.1", "11.9"] },
    { "id": 8, "tasks": ["5.13", "7.8", "9.7", "13.2"] },
    { "id": 9, "tasks": ["11.7", "13.3"] },
    { "id": 10, "tasks": ["13.4", "15"] }
  ]
}
```
