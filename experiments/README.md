# experiments/

Runnable phase scripts only. **No reusable component logic** may be defined
here — import everything from the `src/` package. Each phase script wires
components together, runs experiments, and prints or persists results.

| Script | Phase |
|--------|-------|
| `phase1_retrieval.py`           | Phase 1 — retrieval engine |
| `phase2_reranking.py`           | Phase 2 — cross-encoder reranking |
| `phase3_evaluation.py`          | Phase 3 — baseline vs reranked evaluation |
| `phase4_retrieval_comparison.py`| Phase 4 — strategy & reranker comparison |
| `phase5_end_to_end.py`          | Phase 5 — full local RAG app |
