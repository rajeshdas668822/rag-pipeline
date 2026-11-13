import json, sys
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
from dump_chunks import build_manifest
texts = {}
for line in open("data/chunks_dump.jsonl", encoding="utf-8"):
    if line.strip():
        r = json.loads(line); texts[r["chunk_id"]] = r["text"]
m = build_manifest(texts, 512, 64, ["Mastering RAG 2026.pdf"])
json.dump(m, open("data/chunks_manifest.json", "w", encoding="utf-8"), indent=2)
print(len(texts), m["pypdf_version"])
