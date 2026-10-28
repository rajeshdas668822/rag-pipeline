# RAG Retrieval and Reranking Evaluation Lab

A local-first, educational Python project that measures whether reranking
improves a Retrieval-Augmented Generation (RAG) pipeline. Every component
(chunking, embedding, retrieval, reranking, generation, evaluation) is built
directly in Python — **without LangChain or LlamaIndex** — so the internals are
exposed and independently verifiable.

Embeddings and answer generation run through a locally-running [Ollama](https://ollama.com)
instance; dense similarity is implemented from scratch with NumPy; the primary
reranker is a cross-encoder run locally via Sentence Transformers. No paid APIs.

## Project layout

```
src/           reusable components (loaders, chunker, embedder, retrievers, rerankers, generator, evaluation)
experiments/   runnable phase scripts only — no reusable logic
data/          source documents, evaluation dataset, generated CSV reports
tests/         unit and property-based tests
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"          # core + test tooling
pip install -e ".[dev,optional]" # also install FAISS + Ragas stretch features
```

## Phases

1. Build the retrieval engine from scratch (chunk, embed, index, retrieve).
2. Add a cross-encoder reranker and observe reordering.
3. Evaluate retrieval and reranking scientifically with labelled data.
4. Compare dense / BM25 / hybrid retrieval and cross-encoder vs LLM reranking.
5. Assemble a full local RAG app and judge whether reranking improves answers.

## Ollama runtime

Embeddings, answer generation, and the LLM reranker all talk to a single,
locally-running Ollama instance. You can run Ollama as a **native install** or
in **Docker** — both are equally supported. A native installation is sufficient;
**Docker is not required**. Ollama runs locally only: no cloud or hosted service
is used, and the endpoint must be a locally reachable instance.

Two models are needed:

- `embeddinggemma` — the embedding model
- `qwen2.5:3b` — the chat/instruct model (project default), used by the
  generator and the LLM reranker

### Native install

Install Ollama for your OS, then start the server and pull the models:

```bash
ollama serve                 # serves at http://localhost:11434
ollama pull embeddinggemma
ollama pull qwen2.5:3b
```

With `OLLAMA_ENDPOINT` unset, the default `http://localhost:11434` is used, so
no further configuration is needed.

### Docker

Run Ollama in a container, publishing port 11434 on localhost, then pull the
models inside the container with `docker exec`:

```bash
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec -it ollama ollama pull embeddinggemma
docker exec -it ollama ollama pull qwen2.5:3b
```

Because the container publishes port 11434 on localhost, the default endpoint
works unchanged.

### The `OLLAMA_ENDPOINT` variable

All components (embedder, generator, LLM reranker) read a single endpoint. It
resolves from `OLLAMA_ENDPOINT` → an explicit config value → the default
`http://localhost:11434`. Point it at either the native or containerised
instance — switching runtimes is a configuration change with no code edits:

```bash
# macOS / Linux
export OLLAMA_ENDPOINT=http://localhost:11434
# Windows (PowerShell)
$env:OLLAMA_ENDPOINT = "http://localhost:11434"
```
