"""Corpus manifest drift check shared by the evaluation phases."""

from __future__ import annotations

import hashlib


def compare_manifest(manifest, chunk_map, queries, chunk_size, chunk_overlap):
    """Return (problems, mismatching_ids) comparing the manifest to the current chunks.

    Checks chunk_size/overlap, chunk_count, and the sha256 of the current text of every
    chunk id referenced by the dataset (relevant_documents and graded_labels).
    """
    problems = []
    if manifest.get("chunk_size") != chunk_size or manifest.get("chunk_overlap") != chunk_overlap:
        problems.append(f"chunk config differs: manifest {manifest.get('chunk_size')}/{manifest.get('chunk_overlap')}"
                        f" vs current {chunk_size}/{chunk_overlap}")
    if manifest.get("chunk_count") != len(chunk_map):
        problems.append(f"chunk count differs: manifest {manifest.get('chunk_count')} vs current {len(chunk_map)}")
    referenced = set()
    for q in queries:
        referenced.update(q.relevant_documents)
        referenced.update((q.graded_labels or {}).keys())
    expected = manifest.get("chunk_text_sha256", {})
    mismatched = []
    for cid in sorted(referenced):
        chunk = chunk_map.get(cid)
        if chunk is None or cid not in expected:
            mismatched.append(cid)
            continue
        if hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() != expected[cid]:
            mismatched.append(cid)
    if mismatched:
        problems.append(f"{len(mismatched)} labelled chunk(s) have different text than when labelled")
    return problems, mismatched
