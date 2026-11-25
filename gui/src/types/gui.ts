/**
 * GUI-specific Types
 */

import type { Chunk, RetrievalResult, RerankResult, GenerationResult } from './rag';

export type RetrievalStrategy = 'dense' | 'bm25' | 'hybrid';
export type RerankerType = 'cross-encoder' | 'llm';

export interface RetrievalResultWithText extends RetrievalResult {
  chunk: ChunkWithText;
}

export interface ChunkWithText extends Chunk {
  text: string;
}

export type QueryState = 'idle' | 'loading' | 'success' | 'error';

export interface Metrics {
  recallAt5: number;
  precisionAt5: number;
  ndcgAt5: number;
  mrr: number;
}

export interface QueryResult {
  id: string;
  query: string;
  state: QueryState;
  retrievalResults: RetrievalResultWithText[];
  rerankedResults?: RerankResult[];
  generationResult?: GenerationResult;
  metrics: Metrics;
  error?: string;
}

export interface RAGConfig {
  retrievalStrategy: RetrievalStrategy;
  rerankingEnabled: boolean;
  rerankerType: RerankerType;
  topK: number;
  ollamaEndpoint: string;
  embeddingModel: string;
  instructModel: string;
}
