import json, re

P = "Mastering RAG 2026_0_"
D, M, S, N = "different_terminology", "multi_relevant", "same_topic_different_question", "no_answer"

Q = [
 ("Why did larger context windows not make retrieval obsolete?", {59:3,14:2,19:2}, [M]),
 ("How does cost per query compare between long-context models and RAG?", {19:3,21:3,22:2}, [M,S]),
 ("What is the lost in the middle problem?", {24:3,23:2}, [S]),
 ("What does the needle in a haystack test measure?", {23:3,24:2}, [S]),
 ("When should I fine-tune a model rather than rely on retrieval?", {48:3,47:3,46:2}, [M]),
 ("What are the seven failure points in RAG deployments?", {70:3,71:3,72:3,73:2}, [M]),
 ("What should the system do when a user asks about something the documents never covered?", {70:3,541:3,548:2}, [D,M]),
 ("How do input guardrails protect a RAG system?", {78:3,79:3,80:2,81:2}, [M,S]),
 ("How can personal data in user queries be detected before it reaches the logs?", {78:3,81:1}, [S]),
 ("How should the system handle a follow-up like 'What about the cost?' that lacks context?", {84:3,85:2,86:3}, [D,M]),
 ("How can a request that asks about two things at once be split up before searching?", {87:3,86:2}, [D]),
 ("What is HyDE and why does it help retrieval?", {109:3,57:2}, []),
 ("How does maximal marginal relevance improve the diversity of retrieved results?", {118:3,119:2}, [S]),
 ("What is autocut and how does it filter weak results?", {120:3,119:2}, [S]),
 ("How can I match on small pieces of text but hand the generator the larger enclosing section?", {240:3,121:3}, [D,M]),
 ("How does a cross-encoder reranker score a query against a document?", {463:3,464:3,453:3,452:2}, [M]),
 ("How much latency does reranking add to a RAG pipeline?", {458:3,461:2,467:2,522:2}, [M,S]),
 ("How many comparisons does pairwise reranking need for 50 documents?", {473:3,472:2}, []),
 ("When is LLM listwise reranking worth the several-second wait?", {478:3,479:3,480:2}, [M]),
 ("How does ColBERT late interaction use token embeddings?", {481:3,482:3,480:2,317:2}, [M]),
 ("What are phantom hits in reranking?", {510:3,509:2}, []),
 ("What happens if a document is longer than the reranker can read?", {511:3,512:2,516:1}, [D]),
 ("Why might a reranker trained on Wikipedia fail on medical text?", {513:3,514:3}, [M]),
 ("Which reranker can I self-host on my own hardware with an open license?", {494:3,493:2,499:3,508:1}, [D,M]),
 ("Which reranker supports the longest context window?", {496:3,495:2,498:2}, []),
 ("How should I build a test set to compare rerankers on my own data?", {504:3,337:2,505:1}, [M]),
 ("What does NDCG measure?", {335:3,502:3,501:2}, [M,S]),
 ("How is mean reciprocal rank defined?", {503:3,334:2,335:2}, [M,S]),
 ("How do flat, HNSW, IVF and DiskANN indexes differ?", {387:3,389:3,392:3,390:2,385:2}, [M]),
 ("How much memory does an HNSW index need compared with IVF for 500,000 vectors?", {393:3,388:2,392:2}, []),
 ("How does scalar int8 quantization compare with binary quantization?", {398:3,399:3,395:2}, [M]),
 ("What is the difference between pre-filtering and post-filtering in vector search?", {401:3,402:3,403:1}, [M]),
 ("What does the alpha parameter control in hybrid search?", {312:3,106:3,313:2,405:2}, [M]),
 ("Which embedding approach works best for rare technical terms?", {308:3,309:3,310:1}, []),
 ("Can I shrink embedding vectors to save storage without retraining the model?", {326:3,327:3,325:2}, [D,M]),
 ("How does late chunking differ from the usual chunk-then-embed order?", {228:3,229:3,230:2}, []),
 ("What does context-enriched chunking do and how well does it work?", {224:3,225:3,226:2,223:2}, [M]),
 ("Which chunking strategy suits legal documents and FAQs?", {210:3,209:2,176:2,245:1}, [M]),
 ("What are the drawbacks of semantic splitting?", {215:3,216:3}, [M]),
 ("How do I pick a chunk size based on the type of question users ask?", {185:3,186:3,187:2}, [M]),
 ("How do rerankers reduce hallucinations?", {462:3,460:3,459:2}, [M]),
 ("Why retrieve many candidates first and then rerank only for ordering?", {451:3,453:2,450:2}, [D,M]),
 ("How can I stop one customer's data from showing up in another customer's answers?", {146:3,147:3,148:2}, [D,M]),
 ("How does the system honor a user's request to delete their personal data?", {150:3,151:2,149:1}, [D,M]),
 ("How do I fine-tune a model with LoRA adapters?", {}, [N]),
 ("How do I transcribe audio recordings before indexing them?", {}, [N]),
 ("How do I tune the k1 and b parameters of the BM25 scoring formula?", {}, [N]),
 ("How do I apply stemming and lemmatization before building a TF-IDF index?", {}, [N]),
]

chunks = {}
for l in open("data/chunks_dump.jsonl", encoding="utf-8"):
    r = json.loads(l); chunks[r["chunk_id"]] = r["text"]

def excerpt(cid):
    t = re.sub(r"\s+", " ", chunks[cid])
    t = re.sub(r"-> galileo\.aiMastering Retrieval Augmented Generation RAG", "|", t)
    return t[:300]

out = []
md = ["# Evaluation label review sheet (AI-drafted, pending your review)", "",
 "> **These labels were drafted by an AI assistant from the chunk text and have NOT been human-verified.** Please accept or correct each proposed label.", "",
 "- Chunk config these ids depend on: `chunk_size=512`, `chunk_overlap=64` (Config defaults).",
 "- **Changing the chunk config invalidates every chunk id** below; regenerate with `python experiments/dump_chunks.py` and re-label.",
 "- Inspect any chunk in `data/chunks_dump.jsonl`. Grades: 3 = directly and fully answers, 2 = substantially answers, 1 = partially/tangentially relevant.",
 "- Chunk ids are shortened to `#N` below; full id is `Mastering RAG 2026_0_N`.", ""]
for i, (q, g, tags) in enumerate(Q, 1):
    qid = "q%03d" % i
    rel = [P + str(n) for n in g]
    rec = {"query_id": qid, "query": q, "relevant_documents": rel}
    if g:
        rec["graded_labels"] = {P + str(n): v for n, v in g.items()}
    out.append(json.dumps(rec, ensure_ascii=False))
    md.append("## %s: %s" % (qid, q))
    md.append("Category: %s" % (", ".join(tags) if tags else "single_topic"))
    md.append("")
    if not g:
        md.append("- No relevant chunks (topic absent from the book; verified by keyword search).")
    for n, v in g.items():
        md.append("- `#%d` grade **%d**: %s" % (n, v, excerpt(P + str(n))))
    md.append("")
open("data/evaluation_queries.jsonl", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
open("data/evaluation_labels_review.md", "w", encoding="utf-8").write("\n".join(md))

from collections import Counter
c = Counter(t for _, _, ts in Q for t in ts)
print(len(Q), dict(c), "relevant labels:", sum(len(g) for _, g, _ in Q),
      "distinct chunks:", len({n for _, g, _ in Q for n in g}))
