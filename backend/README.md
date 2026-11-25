# RAG Pipeline Backend

FastAPI backend for the RAG reranking pipeline GUI system.

## Features

- REST API endpoints for RAG operations
- Health check and model listing
- Single query execution with configurable retriever
- Batch query support for evaluation
- Integration with Ollama for embeddings and generation

## API Endpoints

### GET /api/health
Health check endpoint.

### GET /api/models
Lists available Ollama models.

### POST /api/query
Execute a single RAG query.

**Request body:**
```json
{
  "query": "Your question here",
  "retriever_type": "dense",
  "top_k": 50,
  "use_reranking": false,
  "rerank_top_k": 20
}
```

**Response:**
```json
{
  "query_id": "single_query",
  "query": "Your question here",
  "answer": "Generated answer...",
  "citations": ["doc1", "doc2"],
  "insufficient_context": false,
  "retrieval_results": [...],
  "reranked_results": [...],
  "generation_ms": 1234.5,
  "retrieval_ms": 123.4
}
```

### POST /api/batch
Execute batch queries for evaluation.

**Request body:**
```json
{
  "queries": ["Question 1", "Question 2"],
  "query_ids": ["q1", "q2"],
  "retriever_type": "dense",
  "top_k": 50,
  "use_reranking": false
}
```

### GET /api/evaluation
Load evaluation queries from `data/evaluation_queries.jsonl`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama server endpoint |
| `HOST` | `0.0.0.0` | Backend host |
| `PORT` | `8000` | Backend port |

## Local Development

### Prerequisites

- Python 3.11+
- Ollama running locally
- Dependencies from `requirements.txt`

### Installation

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set environment variables
export OLLAMA_ENDPOINT=http://localhost:11434

# Run the server
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Docker Deployment

### Build Images

```bash
# Build all services
docker-compose build

# Or build specific service
docker-compose build backend
docker-compose build gui
```

### Run Services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Volume Structure

```
rag-pipeline/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── rag_service.py
│   ├── Dockerfile
│   └── requirements.txt
├── gui/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
├── data/                    # Documents and evaluation files
│   ├── *.pdf
│   └── evaluation_queries.jsonl
└── docker-compose.yml
```

## Retriever Types

- `dense` - Dense vector retrieval using embeddings
- `bm25` - BM25 lexical retrieval
- `hybrid` - Hybrid of dense + BM25 with RRF fusion

## Reranking

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`) is optional. If it fails to load, the service continues with retrieval-only functionality.
