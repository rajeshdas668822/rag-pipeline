#!/usr/bin/env python
"""Phase 1: Dense retrieval experiment.

This script demonstrates the end-to-end retrieval pipeline:
  Config → OllamaClient → DocumentLoader → Chunker → OllamaEmbedder → DenseRetriever

It accepts command-line arguments for query, chunk size, chunk overlap, and
top-k results. It also runs a naive keyword-based ranking for comparison with
the dense semantic retrieval.

Usage:
    python experiments/phase1_retrieval.py --query "your query here"
    python experiments/phase1_retrieval.py -q "query" --chunk_size 512 --chunk_overlap 64 --top_k 10
"""

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.dense_retriever import DenseRetriever, NumpyBackend, RetrievalResult
from src.document_loader import DocumentLoader
from src.ollama_client import OllamaClient
from src.ollama_embedder import OllamaEmbedder
from src.chunker import Chunker, InvalidChunkConfigError
from src.rag_types import Chunk


def validate_query(query: str) -> None:
    """Validate query input.

    Args:
        query: The query string to validate.

    Raises:
        ValueError: If query is empty or exceeds 2048 characters.
    """
    if not query or len(query) == 0:
        raise ValueError("Query must be non-empty.")
    if len(query) > 2048:
        raise ValueError("Query must not exceed 2048 characters.")


def validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk configuration.

    Args:
        chunk_size: Size of each chunk (must be 50-2048).
        chunk_overlap: Overlap between chunks (must be 0 to chunk_size-1).

    Raises:
        ValueError: If chunk configuration is invalid.
    """
    if chunk_size < 50 or chunk_size > 2048:
        raise ValueError(f"chunk_size must be between 50 and 2048, got {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be between 0 and chunk_size-1, "
            f"got overlap={chunk_overlap} for size={chunk_size}"
        )


def naive_keyword_ranking(query: str, chunks: List[Chunk]) -> List[Tuple[str, float]]:
    """Run a naive keyword-based ranking similar to BM25.

    This implements a simplified TF-IDF style scoring:
    - Tokenize query and chunks into words
    - Score each chunk by summing term frequencies for query terms
    - Penalize chunks that appear in many documents (inverse document frequency)

    Args:
        query: The search query string.
        chunks: List of Chunk objects to rank.

    Returns:
        List of (chunk_id, score) tuples sorted by descending score.
    """
    # Tokenize query into lowercase words
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    if not query_tokens:
        return []

    # Build document frequency dictionary
    doc_freq = Counter()
    chunk_tokens_list = []
    
    for chunk in chunks:
        tokens = re.findall(r'\b\w+\b', chunk.text.lower())
        chunk_tokens_list.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    n_chunks = len(chunks)
    
    # Calculate scores for each chunk
    scores = []
    for i, chunk in enumerate(chunks):
        tokens = chunk_tokens_list[i]
        token_counts = Counter(tokens)
        
        # Simple scoring: sum of (TF * IDF) for query terms
        score = 0.0
        for token in query_tokens:
            tf = token_counts.get(token, 0) / max(len(tokens), 1)
            df = doc_freq.get(token, 0)
            idf = 1.0 + (n_chunks / max(df, 1))
            score += tf * idf
        
        scores.append((chunk.chunk_id, score))

    # Sort by descending score
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


def print_retrieval_results(
    query: str,
    dense_results: List[RetrievalResult],
    keyword_results: List[Tuple[str, float]],
    retrieval_time_ms: float,
    chunks: List[Chunk],
    chunk_map: dict
) -> None:
    """Print retrieval results in a formatted manner.

    Args:
        query: The original query string.
        dense_results: Dense retrieval results with similarity scores.
        keyword_results: Keyword ranking results.
        retrieval_time_ms: Total retrieval time in milliseconds.
        chunks: List of all chunks for content lookup.
        chunk_map: Mapping from chunk_id to Chunk object.
    """
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    print(f"\nQuery: \"{query}\"")
    print(f"Retrieval time: {retrieval_time_ms:.4f} ms\n")

    # Print dense semantic ranking
    print("-" * 80)
    print("DENSE SEMANTIC RANKING (Cosine Similarity)")
    print("-" * 80)
    if dense_results:
        for result in dense_results:
            chunk = chunk_map.get(result.doc_id)
            content_preview = chunk.text[:200] + "..." if chunk and len(chunk.text) > 200 else (chunk.text if chunk else "")
            print(f"Rank {result.rank:2d} | Doc ID: {result.doc_id:20s} | Score: {result.score:.4f}")
            print(f"  Content: {content_preview}")
    else:
        print("No results found.")
    print()

    # Print keyword ranking side by side
    print("-" * 80)
    print("NAIVE KEYWORD RANKING (BM25-like)")
    print("-" * 80)
    if keyword_results:
        for rank, (chunk_id, score) in enumerate(keyword_results, start=1):
            chunk = chunk_map.get(chunk_id)
            content_preview = chunk.text[:200] + "..." if chunk and len(chunk.text) > 200 else (chunk.text if chunk else "")
            print(f"Rank {rank:2d} | Doc ID: {chunk_id:20s} | Score: {score:.4f}")
            print(f"  Content: {content_preview}")
    else:
        print("No results found.")
    print()

    # Summary
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Dense results returned: {len(dense_results)}")
    print(f"Keyword results returned: {len(keyword_results)}")
    
    # Show top dense result doc IDs
    if dense_results:
        dense_ids = [r.doc_id for r in dense_results[:5]]
        print(f"Top dense doc IDs: {dense_ids}")
    
    # Show top keyword result doc IDs
    if keyword_results:
        keyword_ids = [r[0] for r in keyword_results[:5]]
        print(f"Top keyword doc IDs: {keyword_ids}")


def main() -> None:
    """Main entry point for the phase 1 retrieval experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Phase 1: Dense retrieval experiment with keyword comparison"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Search query string"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=64,
        help="Chunk overlap in characters (default: 64)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing documents to index (default: 'data')"
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        validate_query(args.query)
        validate_chunk_config(args.chunk_size, args.chunk_overlap)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Load configuration
    print("Loading configuration...")
    config = Config.load()
    print(f"  Ollama endpoint: {config.ollama_endpoint}")
    print(f"  Embedding model: {config.embedding_model}")

    # Ensure Ollama is ready
    print("\nChecking Ollama availability...")
    ollama_client = OllamaClient(endpoint=config.ollama_endpoint)
    
    try:
        ollama_client.ensure_ready([config.embedding_model])
        print(f"  ✓ Ollama is available and model '{config.embedding_model}' is ready")
    except Exception as e:
        print(f"  ✗ Ollama check failed: {e}")
        sys.exit(1)

    # Load documents
    print(f"\nLoading documents from {args.data_dir}...")
    loader = DocumentLoader(config)
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"  Note: Data directory '{args.data_dir}' does not exist yet")
        print(f"  Please add documents to {data_path.resolve()} before running this script")
        documents = []
    else:
        documents = loader.load_directory(args.data_dir)
        print(f"  Loaded {len(documents)} document(s)")

    # Chunk documents
    print(f"\nChunking documents (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    try:
        chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks = []
        for doc in documents:
            doc_chunks = chunker.chunk(doc)
            chunks.extend(doc_chunks)
        print(f"  Created {len(chunks)} chunks")
    except InvalidChunkConfigError as e:
        print(f"  Chunking failed: {e}")
        sys.exit(1)

    # Embed chunks
    print("\nEmbedding chunks...")
    try:
        embedder = OllamaEmbedder(
            client=ollama_client,
            model=config.embedding_model,
            timeout_s=config.embed_timeout_s
        )
        print(f"  Embedding dimension: {embedder.dimension}")
    except Exception as e:
        print(f"  Embedding setup failed: {e}")
        sys.exit(1)

    # Index with dense retriever
    print("\nIndexing chunks with DenseRetriever...")
    backend = NumpyBackend()
    retriever = DenseRetriever(embedder=embedder, backend=backend)
    
    try:
        retriever.index(chunks)
        print(f"  Indexed {len(chunks)} chunks")
    except Exception as e:
        print(f"  Indexing failed: {e}")
        sys.exit(1)

    # Create chunk map for content lookup
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    # Retrieve results
    print(f"\nRetrieving top {args.top_k} results...")
    start_time = time.perf_counter()
    try:
        retrieval_output = retriever.retrieve(args.query, args.top_k)
    except Exception as e:
        print(f"  Retrieval failed: {e}")
        sys.exit(1)
    
    dense_results = retrieval_output.results
    dense_time_ms = retrieval_output.latency_ms

    # Run keyword ranking for comparison
    print("Running naive keyword ranking...")
    keyword_results = naive_keyword_ranking(args.query, chunks)

    # Calculate total time
    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Print results
    print_retrieval_results(
        query=args.query,
        dense_results=dense_results,
        keyword_results=keyword_results,
        retrieval_time_ms=total_time_ms,
        chunks=chunks,
        chunk_map=chunk_map
    )

    print("\n" + "=" * 80)
    print("END OF RETRIEVAL RESULTS")
    print("=" * 80)


if __name__ == "__main__":
    main()
