# RAG GUI Implementation Tasks

## Phase 1: Project Setup and Foundation

### 1.1 Initialize React Project
- [x] Create new Vite + React project in `gui/` directory
- [-] Configure TypeScript with strict mode
- [~] Set up Tailwind CSS with dark mode support
- [~] Configure basic ESLint/Prettier rules matching project standards
- [~] Create initial directory structure (`src/components`, `src/hooks`, `src/lib`, `src/types`)

### 1.2 Project Configuration
- [~] Set up `package.json` with dependencies:
  - `react`, `react-dom`
  - `vite`, `@vitejs/plugin-react`
  - `tailwindcss`, `postcss`, `autoprefixer`
  - `zustand` (state management)
  - `react-markdown` (answer rendering)
  - `@heroicons/react` (icons)
  - `chart.js`, `react-chartjs-2` (metrics visualization)
- [~] Configure Vite proxy for Ollama API (localhost:11434)
- [~] Configure Vite proxy for backend API (localhost:8000)
- [~] Set up `vite.config.ts` with path aliases

### 1.3 Type Definitions
- [~] Create `src/types/rag.ts` matching Python `rag_types.py`:
  - `Document`, `Chunk`, `RetrievalResult`, `RerankResult`, `GenerationResult`
- [~] Create `src/types/gui.ts`:
  - `RetrievalStrategy` (dense | bm25 | hybrid)
  - `RerankerType` (cross-encoder | llm)
  - `RetrievalResultWithText` (includes chunk text)
  - `QueryState` (idle | loading | success | error)
  - `Metrics` (recall@5, precision@5, ndcg@5, mrr)
- [~] Export types from `src/types/index.ts`

## Phase 1.5: Backend API Integration

### 1.5.1 API Client Service
- [ ] Create `src/lib/apiClient.ts` with:
  - `axios` instance with base URL from config
  - Request/response interceptors for logging
  - Automatic error handling with user-friendly messages
  - Token refresh support (if needed)
- [ ] Implement `get /api/health` endpoint
- [ ] Implement `get /api/models` endpoint
- [ ] Implement `post /api/query` endpoint
- [ ] Implement `post /api/batch` endpoint
- [ ] Implement `get /api/evaluation` endpoint
- [ ] Add request/response logging interceptors
- [ ] Add retry logic for transient failures (max 3 retries)
- [ ] Test API client with backend server

### 1.5.2 API Configuration
- [ ] Add `BACKEND_API_URL` environment variable support
- [ ] Add fallback to Ollama URL if backend unavailable
- [ ] Add API version negotiation endpoint

### 1.5.3 Caching
- [ ] Implement in-memory cache for models list (TTL: 60s)
- [ ] Implement cache invalidation on model changes
- [ ] Add cache hit/miss metrics

## Phase 2: Core Infrastructure

### 2.1 Ollama Client (Legacy)
- [ ] Update `src/lib/ollamaClient.ts` to use API endpoints instead of direct calls
- [ ] Migrate `getModels()` to use `/api/models`
- [ ] Migrate `generateEmbedding()` to use `/api/query`
- [ ] Migrate `generateAnswer()` to use `/api/query`
- [ ] Deprecate direct Ollama endpoint calls

### 2.2 RAG Service Layer
- [ ] Update `src/lib/ragService.ts` to use backend API instead of direct RAG calls
- [ ] Replace `retrieveDense()`, `retrieveBM25()`, `retrieveHybrid()` with single `/api/query` call
- [ ] Replace `rerankCrossEncoder()`, `rerankLLM()` with backend processing
- [ ] Replace `generateAnswer()` with backend answer generation
- [ ] Maintain local metrics calculation for display
- [ ] Handle fallback mode when backend unavailable

### 2.3 Configuration Management
- [ ] Create `src/store/configStore.ts` with Zustand:
  - Retrieval strategy state
  - Reranking enabled/disabled
  - Reranker type
  - Top-K value
  - Backend API endpoint (NEW)
  - Ollama endpoint (legacy fallback)
  - Selected models (embedding, instruct)
- [ ] Implement localStorage persistence
- [ ] Add validation and default values

### 2.4 Data Loading
- [ ] Create `src/lib/dataLoader.ts`:
  - `loadEvaluationQueries()` - parse `data/evaluation_queries.jsonl`
  - `loadEvaluationLabels()` - load reference documents
  - `loadChunksManifest()` - load chunk metadata from `data/chunks_manifest.json`
- [ ] Implement error handling for missing files

## Phase 3: UI Components

### 3.1 Layout Components
- [ ] `src/components/layout/AppLayout.tsx` - main layout with sidebar + main area
- [ ] `src/components/layout/Sidebar.tsx` - configuration panel
- [ ] `src/components/layout/TopBar.tsx` - title, model selector, clear/reset buttons, **API status indicator**
- [ ] `src/components/layout/StatusBar.tsx` - connection status, metrics summary

### 3.2 Input Components
- [ ] `src/components/query/QueryInput.tsx` - text input with submit/clear
- [ ] `src/components/query/QueryHistory.tsx` - recent queries list
- [ ] `src/components/query/ConfigurationPanel.tsx` - strategy, reranking, top-K controls

### 3.3 Result Display Components
- [ ] `src/components/results/RetrievalResultItem.tsx` - single result card
- [ ] `src/components/results/RetrievalResultsList.tsx` - list of retrieved documents
- [ ] `src/components/results/ChunkViewer.tsx` - modal to view full chunk text
- [ ] `src/components/answer/AnswerDisplay.tsx` - rendered answer with citations
- [ ] `src/components/answer/AnswerMetrics.tsx` - recall@5, precision@5, NDCG@5, MRR

### 3.4 Comparison Components
- [ ] `src/components/comparison/SideBySideComparison.tsx` - baseline vs reranked panels
- [ ] `src/components/comparison/ResultDiffIndicator.tsx` - rank change arrows
- [ ] `src/components/comparison/ComparisonMetrics.tsx` - delta metrics

### 3.5 Metrics Components
- [ ] `src/components/metrics/MetricsBar.tsx` - horizontal metrics visualization
- [ ] `src/components/metrics/PerQueryTable.tsx` - detailed per-query results table
- [ ] `src/components/metrics/SummaryMetrics.tsx` - batch summary statistics

## Phase 4: State Management and Hooks

### 4.1 Query State Hook
- [ ] Create `src/hooks/useQuery.ts`:
  - Query input handling
  - Execution state (idle/loading/error)
  - Results storage
  - Error state management
- [ ] Implement debouncing for input
- [ ] Auto-execute on Enter key

### 4.2 RAG Pipeline Hook
- [ ] Create `src/hooks/useRAGPipeline.ts`:
  - Call RAG service with current config
  - Manage execution lifecycle
  - Store intermediate results (retrieved, reranked, answer)
  - Calculate metrics
- [ ] Implement abort signal for cancellation

### 4.3 Comparison Hook
- [ ] Create `src/hooks/useComparison.ts`:
  - Run baseline retrieval (no reranking)
  - Run reranked retrieval
  - Compare results
  - Calculate delta metrics
- [ ] Optimize to skip redundant calls

## Phase 5: Data Display Logic

### 5.1 Result Mapping
- [ ] Implement `mapRetrievalResultsWithText()` - merge retrieval results with chunk text
- [ ] Handle missing chunks gracefully (show warning, use placeholder)
- [ ] Create `mapRerankResults()` - format reranking output

### 5.2 Answer Formatting
- [ ] Implement `extractCitations(answerText)` - find [doc_id] references
- [ ] Implement `highlightCitations()` - make citations clickable
- [ ] Implement `checkInsufficientContext()` - detect error responses

### 5.3 Metrics Calculation
- [ ] Implement `calculateRecallAtK(results, relevantDocs, k)`
- [ ] Implement `calculatePrecisionAtK(results, relevantDocs, k)`
- [ ] Implement `calculateNDCGAtK(results, relevantDocs, k)`
- [ ] Implement `calculateMRR(results, relevantDocs)`

## Phase 6: Side-by-Side Comparison

### 6.1 UI Implementation
- [ ] Implement comparison mode toggle
- [ ] Create side-by-side layout component
- [ ] Add visual indicators for rank changes (↑↓→)
- [ ] Sync scrolling between panels

### 6.2 Data Preparation
- [ ] Implement `prepareComparisonData()` - format both runs for display
- [ ] Calculate and display metric deltas
- [ ] Show document overlap statistics

## Phase 7: Evaluation Features

### 7.1 Dataset Loading UI
- [ ] Create `src/components/evaluation/LoadEvaluationButton.tsx`
- [ ] Implement progress indicator for batch loading
- [ ] Show loaded query count

### 7.2 Batch Processing
- [ ] Create `src/components/evaluation/BatchRunButton.tsx`
- [ ] Implement sequential execution with progress
- [ ] Add batch cancel option

### 7.3 Results Summary
- [ ] Implement summary statistics calculation
- [ ] Create evaluation results export (JSON)
- [ ] Add per-query table with expandable details

## Phase 8: Polish and User Experience

### 8.1 Loading States
- [ ] Implement skeleton loaders for results
- [ ] Add progress indicators for API calls
- [ ] Show partial results as they arrive (streaming if possible)
- [ ] Add loading states for API health check and model list fetch

### 8.2 Error Handling
- [ ] Create `src/components/errors/OllamaConnectionError.tsx`
- [ ] Create `src/components/errors/ApiConnectionError.tsx`
- [ ] Implement retry logic for failed requests
- [ ] Add helpful error messages with troubleshooting steps
- [ ] Add API-specific error codes and messages
- [ ] Implement error boundary for API failures

### 8.3 User Feedback
- [ ] Add toast notifications for key events
- [ ] Implement undo for recent actions
- [ ] Add keyboard shortcuts (Enter to submit, ESC to close)

## Phase 9: Deployment & Docker

### 9.0 GUI Testing and Documentation
- [ ] Test UI components in isolation
- [ ] Test event handlers
- [ ] Test error states
- [ ] Update `README.md` with GUI usage instructions
- [ ] Document configuration options
- [ ] Add troubleshooting section
- [ ] Create user guide for common tasks

### 9.1 Backend Docker Build
- [ ] Verify Docker build for GUI
- [ ] Verify Docker build for Backend
- [ ] Verify `docker-compose.yml` configuration
- [ ] Test end-to-end with Docker
- [ ] Document Docker deployment in README

### 9.2 Docker Environment
- [ ] Document environment variables for backend API
- [ ] Add health check endpoint to Docker healthcheck
- [ ] Document port mappings (GUI: 5173, Backend: 8000)

## Implementation Notes

### Execution Order
1. Phase 1: Project setup (foundational)
2. Phase 1.5: Backend API integration (new - API client, endpoints)
3. Phase 2: Core infrastructure (update to use backend API)
4. Phase 3: UI components (add API status, loading states)
5. Phase 4: State management (connect UI to API)
6. Phase 5-7: Feature implementation (core functionality)
7. Phase 8: Polish and error handling (API errors)
8. Phase 9: Deployment & Docker (Docker builds, testing)

### Parallel Work
- Phase 1.5 (API client) and Phase 2 (backend integration) can proceed in parallel
- Phase 3 (components) and Phase 4 (hooks) can proceed in parallel
- Phase 5-7 (features) can be iterated incrementally
- Phase 9 (Docker) should run after Phase 1.5 completion

### Key Decisions
- Use Zustand for state (lightweight, minimal boilerplate)
- Local storage for config persistence
- Backend API with fallback to direct Ollama calls
- Client-side rendering only (SSR not needed)
- API caching with 60s TTL for models list
- Automatic retry with exponential backoff for transient failures
