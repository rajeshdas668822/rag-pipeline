# RAG GUI Requirements

## Overview

Create a lightweight, ChatGPT-style single-page application (SPA) to verify and interact with the RAG pipeline developed in the rag-pipeline project. The GUI provides an interactive interface for testing retrieval strategies, reranking options, and answer generation via a FastAPI backend that communicates with Ollama.

### Non-Goals
- Multi-user support or session persistence
- Document upload/corpus management
- Production-grade deployment
- Backend API development (handled by separate backend team)

## User Goals

1. **Query Testing**: Enter natural language questions and see how the RAG system responds
2. **Strategy Comparison**: Compare dense, BM25, and hybrid retrieval strategies
3. **Reranking Verification**: Toggle cross-encoder and LLM reranking to see impact on results
4. **Transparency**: View intermediate results (retrieved documents, scores, reranking changes)
5. **Answer Verification**: See generated answers with citations and assess quality
6. **Performance Insight**: View latency metrics for each component

## Functional Requirements

### FR1. Query Input
- Single-line text input for questions
- Optional submit button (Enter key supported)
- Clear/refresh button to reset current query
- Recent queries history (last 10-20)

### FR2. Configuration Panel
- **Retrieval Strategy Selector**: Radio buttons or dropdown with options:
  - Dense (vector similarity)
  - BM25 (keyword-based)
  - Hybrid (RRF fusion)
- **Reranking Toggle**: Switch to enable/disable reranking
- **Reranker Type**: When reranking is enabled, selector between:
  - Cross-encoder (sentence-transformers)
  - LLM reranker (Ollama)
- **Top-K Slider**: Number of documents to retrieve (1-20, default 5)
- **Model Selector**: Choose between available Ollama models

### FR3. Retrieval Display
- **Retrieved Documents Panel**: List of documents with:
  - Document ID
  - Relevance score
  - Rank position
  - Chunk text preview (first 200-300 chars)
  - Expand button to show full chunk text
- **Before/After Reranking View**: When reranking is enabled:
  - Left column: Original retrieval results (score, rank)
  - Right column: Reranked results (rerank_score, new rank)
  - Visual indicator of rank changes (arrow icons)

### FR4. Answer Display
- **Answer Area**: Markdown-rendered answer text
- **Citations**: Highlighted references with document IDs in brackets [doc_id]
- **Insufficient Context Warning**: Display when answer is "cannot be determined"
- **Latency Breakdown**: Show timing for:
  - Embedding (ms)
  - Retrieval (ms)
  - Reranking (ms)
  - Generation (ms)
  - Total (ms)

### FR5. Metrics Visualization
- **Current Query Metrics**: Display below answer:
  - Recall@5: X%
  - Precision@5: X%
  - NDCG@5: X.XX
  - MRR: X.XX
- **Evaluation Mode**: When loaded with evaluation queries:
  - Batch processing button
  - Summary metrics (avg Recall@5, NDCG@5, etc.)
  - Per-query results table

### FR6. Side-by-Side Comparison
- **Compare Toggle**: Button to enable comparison mode
- **Dual View**: Two panels side by side:
  - Left: Baseline (retrieval only, no reranking)
  - Right: Reranked (retrieval + reranking)
- **Differences Highlighted**:
  - Rank changes in retrieved documents
  - Score changes
  - Different documents in top-K
- **Comparison Metrics**: Show delta for key metrics

### FR7. Evaluation Data Loading
- **Load Evaluation Dataset**: Button to load queries from data/evaluation_queries.jsonl
- **Load Labels**: Reference relevant documents for each query
- **Status Indicator**: Show loaded query count and available metrics

### FR8. State Management
- **Persist Configuration**: Save user preferences (strategy, rerank toggle, top-K) in localStorage
- **Reset State**: Clear button to reset all settings to defaults

## Technical Requirements

### TR1. API-Based Architecture
The GUI communicates with a FastAPI backend service that orchestrates the RAG pipeline:

**Three-Tier Architecture:**
```
Browser (GUI) → Backend API → Ollama
```

- **Browser Layer**: React-based SPA handling UI, state, and user interactions
- **Backend API Layer**: FastAPI server that runs the RAG pipeline logic
- **Ollama Layer**: Local LLM service for embeddings and text generation

All computation (embedding, retrieval, reranking, generation) happens on the backend. The GUI only handles display and user input.

### TR2. Browser Compatibility
- Modern browsers (Chrome, Firefox, Edge, Safari)
- ES2020+ features
- Fetch API or axios for HTTP requests

### TR3. Performance
- GUI cold start < 2 seconds
- Backend API response < 5 seconds for typical retrieval (k=5)
- Smooth UI during backend processing

### TR4. Error Handling
- API connection errors displayed clearly
- Backend service unavailable messages
- Model not found errors with helpful instructions
- Graceful degradation when reranking fails
- Timeout handling for slow API responses

## API Endpoints

The GUI consumes the following FastAPI endpoints:

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| GET | `/api/health` | Backend health check | None | `{status: "healthy"}` |
| GET | `/api/models` | List available Ollama models | None | `{models: ["model1", "model2"]}` |
| POST | `/api/query` | Single query with full RAG pipeline | `{query: string, strategy: string, rerank: boolean, rerankerType: string, topK: number, model: string}` | Full RAG result with documents, answer, latency |
| POST | `/api/batch` | Batch evaluation processing | `{queries: Query[], strategy: string, rerank: boolean, rerankerType: string, topK: number, model: string}` | `{results: [], summary: {}}` |
| GET | `/api/evaluation` | Load evaluation dataset | None | `{queries: [], labels: {}}` |

### Request/Response Formats

**Query Request:**
```json
{
  "query": "What is the capital of France?",
  "strategy": "hybrid",
  "rerank": true,
  "rerankerType": "llm",
  "topK": 5,
  "model": "qwen2.5:3b"
}
```

**Query Response:**
```json
{
  "query": "What is the capital of France?",
  "documents": [
    {
      "id": "doc_1",
      "score": 0.95,
      "rank": 1,
      "text": "..."
    }
  ],
  "answer": "The capital of France is Paris.",
  "citations": ["doc_1", "doc_3"],
  "latency": {
    "embedding": 150,
    "retrieval": 200,
    "reranking": 300,
    "generation": 800,
    "total": 1450
  }
}
```

**Error Response:**
```json
{
  "error": "Backend service unavailable",
  "message": "Failed to connect to backend API at http://localhost:8000",
  "statusCode": 503
}
```

## Data Flow

1. User enters query and selects options in the GUI
2. GUI sends POST `/api/query` to backend API with configuration
3. Backend receives request and:
   - Calls Ollama for embedding (if dense/hybrid)
   - Retrieves documents via configured strategy
   - Reranks if enabled (cross-encoder or LLM)
   - Sends context + query to Ollama for generation
4. Backend returns complete RAG result to GUI
5. GUI displays answer, documents, citations, and metrics

## Authentication & Security

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:8000` |
| `VITE_API_TIMEOUT` | Request timeout in ms | `30000` |

### CORS Considerations
- Backend must enable CORS for GUI origin (e.g., `http://localhost:5173`)
- Backend should specify allowed methods: `GET, POST, OPTIONS`
- Backend should include: `Content-Type, Authorization`

### Error Response Handling
- 4xx errors: Display user-friendly message with details
- 5xx errors: Show service unavailable message
- Network errors: Display connection failure with retry option
- Timeouts: Show "request timed out" message

### Security Considerations
- API URL must be configurable via environment variables
- No hardcoded API endpoints in code
- Sensitive data should not be logged in browser console
- Error messages should not expose backend internals

## Dependencies

### External Services
- Backend API (FastAPI, typically http://localhost:8000)
- Ollama (running on backend, http://localhost:11434)

### Libraries
- React 18+ for UI
- TypeScript for type safety
- Tailwind CSS for styling
- Zustand or Redux Toolkit for state management
- React Markdown for answer rendering with syntax highlighting
- Plotly or Chart.js for metrics visualization
- axios or native fetch API for HTTP requests

## User Interface Requirements

### UI1. Layout
- **Top Bar**: Application title, model selector, clear/reset buttons
- **Sidebar**: Configuration panel (strategy, reranking, top-K)
- **Main Area**: Query input, result display
- **Status Bar**: Connection status, last update time

### UI2. ChatGPT-Style Interface
- Clean, minimal aesthetic
- Dark theme default with light theme option
- Syntax highlighting for code blocks in answers
- Copy to clipboard for answers and citations

### UI3. Feedback
- Loading indicators during API calls
- Progress bar for batch operations
- Toast notifications for success/error messages
- Connection status indicator (online/offline)

## Acceptance Criteria

1. **API Connection**: GUI successfully connects to backend API and retrieves models list
2. **Query Testing**: User can enter a question and see a generated answer within 10 seconds
3. **Strategy Switching**: Changing retrieval strategy updates results within 5 seconds
4. **Reranking Toggle**: Enabling/disabling reranking visibly changes document rankings
5. **Citation Display**: All citations in answer are correctly linked to document IDs
6. **Metrics Calculation**: Metrics match expected values from evaluation dataset
7. **Side-by-Side Comparison**: Comparison view clearly shows differences between baseline and reranked
8. **Error Resilience**: System handles API connection errors gracefully without crashing
9. **Timeout Handling**: Requests timeout after configured duration with appropriate error message

## Out of Scope

- User authentication
- Multi-document corpus management
- Advanced filtering/sorting of results
- Export to file (PDF, CSV)
- History persistence beyond session
- Mobile optimization
- Backend API development (handled separately)