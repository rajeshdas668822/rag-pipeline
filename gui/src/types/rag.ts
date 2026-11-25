/**
 * RAG Types - matches Python rag_types.py
 */

export interface Document {
  id: string;
  title: string;
  content: string;
  metadata: Record<string, unknown>;
}

export interface Chunk {
  id: string;
  documentId: string;
  text: string;
  embedding?: number[];
  score?: number;
  rank?: number;
  metadata: Record<string, unknown>;
}

export interface RetrievalResult {
  chunk: Chunk;
  score: number;
  source: 'dense' | 'bm25' | 'hybrid';
}

export interface RerankResult {
  chunk: Chunk;
  score: number;
  originalRank: number;
  newRank: number;
}

export interface GenerationResult {
  answer: string;
  chunks: Chunk[];
  metadata: {
    model: string;
    promptTokens: number;
    completionTokens: number;
    latencyMs: number;
  };
}
