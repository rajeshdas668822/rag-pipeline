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
backend/       FastAPI REST API for RAG operations
gui/           React application for RAG interface
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

## Docker Deployment

The project includes a complete Docker setup with FastAPI backend and React GUI.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Browser                                   │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                             │
│                    │     GUI (React)  │                             │
│                    │   Port 5173      │                             │
│                    └────────┬─────────┘                             │
│                             │                                       │
│                             │ HTTP requests                         │
│                             │                                       │
│                    ┌────────▼─────────┐                             │
│                    │   Backend        │                             │
│                    │  (FastAPI)       │                             │
│                    │   Port 8000      │                             │
│                    └────────┬─────────┘                             │
│                             │                                       │
│                             │ Python RAG Logic                      │
│                             │ - Retrievers (Dense/BM25/Hybrid)      │
│                             │ - Cross-encoder Reranker              │
│                             │ - Ollama Generator                    │
│                             ▼                                       │
│                    ┌──────────────────┐                             │
│                    │    Ollama        │                             │
│                    │  Port 11434      │                             │
│                    │  (Embeddings +   │                             │
│                    │   Generation)    │                             │
│                    └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

- **Backend (FastAPI)**: Python service running on port 8000 that exposes RAG
  functionality via REST API. Implements retrieval (dense, BM25, hybrid),
  reranking (cross-encoder, LLM), and answer generation via Ollama.

- **GUI (React)**: Single-page application running on port 5173 that provides a
  web interface for querying the RAG pipeline. Connects to the backend API
  using `VITE_API_URL=http://localhost:8000`.

- **Ollama**: Containerized LLM service running on port 11434 that provides
  embeddings and text generation. The backend and Ollama communicate over the
  Docker network (`http://ollama:11434`).

### Quick Start with Docker

1. **Ensure Docker is running**:

   ```bash
   docker --version
   ```

2. **Start all services**:

   ```bash
   docker-compose up -d
   ```

   This command:
   - Starts Ollama and pulls the required models (`embeddinggemma`, `qwen2.5:3b`)
   - Builds and starts the FastAPI backend
   - Builds and starts the React GUI

3. **Wait for Ollama to be ready** (about 30-60 seconds):

   ```bash
   docker-compose logs ollama
   ```

   Look for `ollama service is ready` message.

4. **Access the GUI**:

   Open `http://localhost:5173` in your browser.

5. **Stop services**:

   ```bash
   docker-compose down
   ```

   To also remove volumes (embedded models):

   ```bash
   docker-compose down -v
   ```

### How the GUI Connects to the Backend

The React GUI uses environment variables to configure the API endpoint:

- `VITE_API_URL=http://localhost:8000` is set in `docker-compose.yml`
- The GUI's `ragService.ts` makes HTTP requests to this endpoint
- The `ollamaClient.ts` connects directly to Ollama at `http://localhost:11434`

For API endpoints, the GUI calls:
- `GET /api/health` - Health check
- `GET /api/models` - List available Ollama models
- `POST /api/query` - Execute a single RAG query
- `POST /api/batch` - Execute batch queries
- `GET /api/evaluation` - Load evaluation queries

### Backend Directory Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── models.py            # Pydantic models for requests/responses
├── rag_service.py       # RAG orchestration logic
├── Dockerfile           # Backend container image
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variable template
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://ollama:11434` (Docker) | Ollama server endpoint |
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL for GUI |

## Local Development

Run the backend API and GUI locally (without Docker):

### Backend

```bash
# Install dependencies
pip install -e ".[dev]"

# Set environment variables
export OLLAMA_ENDPOINT=http://localhost:11434

# Run the server
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### GUI

```bash
cd gui
npm install
npm run dev
```

The GUI will be available at `http://localhost:5173`.

### Running Experiments

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Run phase scripts
python experiments/phase1_retrieval.py
python experiments/phase2_reranking.py
python experiments/phase3_evaluation.py
python experiments/phase4_retrieval_comparison.py
python experiments/phase5_end_to_end.py
```

## Retriever Types

- `dense` - Dense vector retrieval using embeddings
- `bm25` - BM25 lexical retrieval
- `hybrid` - Hybrid of dense + BM25 with RRF fusion

## Reranking

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`) is optional. If it fails to load, the service continues with retrieval-only functionality.
