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

## Running on a Different Machine

This section covers deploying the RAG pipeline on a remote machine (e.g., AWS EC2, distant server) where Ollama, the backend, and GUI may run on separate machines or require network access.

### Prerequisites

- **Docker installed** on the target machine (if using Docker for Ollama)
- **Ollama models available** on the remote machine (`embeddinggemma`, `qwen2.5:3b`)
- **Network access** between the GUI, backend, and Ollama machines (if distributed)
- **Python 3.11+** if running experiments locally without Docker

### Setup Scenarios

#### Scenario 1: All Services on One Remote Machine

This is the simplest remote setup—everything runs on the same server.

1. **Copy the repository** to the remote machine:
   ```bash
   git clone <your-repo-url> rag-pipeline
   cd rag-pipeline
   ```

2. **Configure Ollama for remote access** (if using Docker):
   
   Edit `ollama/docker-compose.yml` to bind to all interfaces:
   ```yaml
   services:
     ollama:
       ports:
         - "0.0.0.0:11434:11434"  # Bind to all interfaces
   ```

   Then start Ollama:
   ```bash
   cd ollama
   docker-compose up -d
   docker exec -it ollama ollama pull embeddinggemma
   docker exec -it ollama ollama pull qwen2.5:3b
   ```

3. **Configure the backend** to use the remote Ollama endpoint:
   
   Create a `.env` file in the `backend/` directory:
   ```bash
   cd ../backend
   cp .env.example .env
   ```
   
   Edit `.env` to use the server's IP address:
   ```env
   OLLAMA_ENDPOINT=http://<server-ip>:11434
   ```

4. **Start the backend**:
   ```bash
   docker-compose up -d
   ```

5. **Configure the GUI** to connect to the backend API:
   
   The `docker-compose.yml` already sets `VITE_API_URL=http://localhost:8000`, which works when the GUI and backend are on the same machine. For a remote backend, update the `gui/.env` file:
   ```bash
   cd ../gui
   echo "VITE_API_URL=http://<backend-ip>:8000" > .env
   npm run build
   ```
   
   Then deploy the built files (e.g., via nginx) or run the development server:
   ```bash
   npm run dev -- --host 0.0.0.0 --port 5173
   ```

#### Scenario 2: Distributed Setup (Ollama, Backend, GUI on Separate Machines)

For maximum flexibility, run each service on a dedicated machine.

**Machine A: Ollama Server**
1. Install Ollama (native or Docker)
2. Pull required models:
   ```bash
   ollama pull embeddinggemma
   ollama pull qwen2.5:3b
   ```
3. If using Docker, configure `0.0.0.0:11434:11434` in `docker-compose.yml`
4. Note the machine's IP address: `OLLAMA_SERVER_IP`

**Machine B: Backend Server**
1. Copy the repository
2. Create `backend/.env`:
   ```env
   OLLAMA_ENDPOINT=http://<OLLAMA_SERVER_IP>:11434
   ```
3. Start the backend (Docker or local):
   ```bash
   # Docker
   docker-compose up -d backend
   
   # Or local
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
4. Note the machine's IP address: `BACKEND_SERVER_IP`

**Machine C: GUI Server**
1. Copy the repository
2. Configure the GUI to connect to the remote backend:
   ```bash
   cd gui
   echo "VITE_API_URL=http://<BACKEND_SERVER_IP>:8000" > .env
   npm run dev -- --host 0.0.0.0 --port 5173
   ```

### Firewall/Network Requirements

- **Ollama port (11434)**: Must be open between backend and Ollama machines
- **Backend port (8000)**: Must be open between GUI and backend machines
- **GUI port (5173)**: Must be open for browser access (or use a reverse proxy like nginx)

Example `ufw` rules on Linux:
```bash
# Allow Ollama from backend machine only
ufw allow from <backend-ip> to any port 11434

# Allow backend from GUI machine only
ufw allow from <gui-ip> to any port 8000

# Allow GUI from anywhere (or restrict to your IP)
ufw allow 5173
```

### Testing the Remote Setup

1. **Verify Ollama is accessible**:
   ```bash
   curl http://<ollama-server-ip>:11434/api/tags
   # Should return list of models
   ```

2. **Verify backend is accessible**:
   ```bash
   curl http://<backend-server-ip>:8000/api/health
   # Should return {"status":"healthy"}
   ```

3. **Test GUI connectivity**:
   - Open browser to `http://<gui-server-ip>:5173`
   - Try a sample query to verify full pipeline works

### Running Phase 5 with Remote Ollama

When running `phase5_end_to_end.py` with Ollama on a remote machine:

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Run with --ollama_endpoint pointing to remote server
python experiments/phase5_end_to_end.py \
  --ollama_endpoint http://<OLLAMA_SERVER_IP>:11434 \
  --data_dir data \
  --dataset data/evaluation_queries.jsonl \
  --out_dir data
```

The `--ollama_endpoint` flag overrides the default `http://localhost:11434` and sets the endpoint for:
- Embedding generation (`OllamaEmbedder`)
- Answer generation (`OllamaGenerator`)
- Model readiness checks (`OllamaClient.ensure_ready`)

All components respect the `Config.load()` precedence:
1. `--ollama_endpoint` flag → `overrides={"ollama_endpoint": "..."}`
2. `OLLAMA_ENDPOINT` environment variable
3. Default `http://localhost:11434`

### Troubleshooting

**Ollama not reachable from backend**
- Verify Ollama is running: `curl http://<ollama-ip>:11434/api/tags`
- Check firewall rules
- Ensure Ollama is bound to `0.0.0.0` (not just `localhost`)

**Backend not reachable from GUI**
- Check backend logs: `docker-compose logs backend`
- Verify CORS is enabled (it is by default in `backend/main.py`)
- Test with `curl http://<backend-ip>:8000/api/models`

**Phase 5 fails with connection errors**
- Verify Ollama endpoint: `python -c "from src.ollama_client import OllamaClient; c = OllamaClient(endpoint='http://<ip>:11434'); c.ensure_ready(['embeddinggemma', 'qwen2.5:3b'])"`
- Check that the models exist: `curl http://<ollama-ip>:11434/api/tags`
