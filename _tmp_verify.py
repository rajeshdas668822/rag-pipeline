import sys, json
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
from src.cross_encoder_reranker import CrossEncoderReranker
from src.rag_types import Chunk
from src.evaluation_dataset import load_evaluation_queries
import phase3_evaluation as p3

# index()
c = [Chunk(chunk_id=f"c{i}", doc_id="d", text=f"text {i}", seq=i) for i in range(3)]
for inp in ({x.chunk_id: x for x in c}, {x.chunk_id: x.text for x in c}, c):
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r.index(inp)
    assert r._texts == {"c0": "text 0", "c1": "text 1", "c2": "text 2"}, r._texts
print("index ok")

# manifest comparison
rows = [json.loads(l) for l in open("data/chunks_dump.jsonl", encoding="utf-8") if l.strip()]
cmap = {r["chunk_id"]: Chunk(chunk_id=r["chunk_id"], doc_id="d", text=r["text"], seq=0) for r in rows}
manifest = json.load(open("data/chunks_manifest.json", encoding="utf-8"))
qs = load_evaluation_queries("data/evaluation_queries.jsonl", corpus_ids=set(cmap))
probs, mm = p3.compare_manifest(manifest, cmap, qs, 512, 64)
assert not probs and not mm, probs
print("match passes; queries:", len(qs))
# differing text for a referenced id
ref = next(d for q in qs for d in q.relevant_documents)
bad = dict(cmap); bad[ref] = Chunk(chunk_id=ref, doc_id="d", text="different", seq=0)
probs, mm = p3.compare_manifest(manifest, bad, qs, 512, 64)
assert mm == [ref] and probs, (probs, mm)
# different count (715-like) and config
bad2 = dict(cmap); bad2["extra"] = cmap[ref]
probs, mm = p3.compare_manifest(manifest, bad2, qs, 512, 64)
assert any("count" in x for x in probs)
probs, mm = p3.compare_manifest(manifest, cmap, qs, 256, 64)
assert any("config" in x for x in probs)
print("mismatch detection ok")
