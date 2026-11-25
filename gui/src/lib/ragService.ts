import type { Chunk, GenerationResult, Metrics, RerankResult, RetrievalResult } from '../types';
import { ollamaClient } from './ollamaClient';

interface RAGServiceConfig {
  topK?: number;
  embeddingModel?: string;
  instructModel?: string;
}

class RAGService {
  private config: RAGServiceConfig;

  constructor(config: RAGServiceConfig = {}) {
    this.config = {
      topK: config.topK ?? 10,
      embeddingModel: config.embeddingModel ?? 'nomic-embed-text',
      instructModel: config.instructModel ?? 'mistral',
    };
  }

  async retrieveDense(query: string, _k: number = 10): Promise<RetrievalResult[]> {
    try {
      await ollamaClient.generateEmbedding(this.config.embeddingModel!, query);
      // In a real implementation, this would query a vector database
      // For now, return placeholder results
      return [
        {
          chunk: {
            id: 'chunk-1',
            documentId: 'doc-1',
            text: 'Sample document content for dense retrieval.',
            metadata: {},
          },
          score: 0.95,
          source: 'dense',
        },
        {
          chunk: {
            id: 'chunk-2',
            documentId: 'doc-2',
            text: 'Another sample document for dense retrieval.',
            metadata: {},
          },
          score: 0.87,
          source: 'dense',
        },
      ] as RetrievalResult[];
    } catch (error) {
      console.error('Dense retrieval failed:', error);
      return [];
    }
  }

  async retrieveBM25(_query: string, _k: number = 10): Promise<RetrievalResult[]> {
    try {
      // In a real implementation, this would query a BM25 index
      return [
        {
          chunk: {
            id: 'chunk-3',
            documentId: 'doc-3',
            text: 'Sample document content for BM25 retrieval.',
            metadata: {},
          },
          score: 15.2,
          source: 'bm25',
        },
        {
          chunk: {
            id: 'chunk-4',
            documentId: 'doc-4',
            text: 'Another sample document for BM25 retrieval.',
            metadata: {},
          },
          score: 12.8,
          source: 'bm25',
        },
      ] as RetrievalResult[];
    } catch (error) {
      console.error('BM25 retrieval failed:', error);
      return [];
    }
  }

  async retrieveHybrid(query: string, k: number = 10): Promise<RetrievalResult[]> {
    try {
      const denseResults = await this.retrieveDense(query, k);
      const bm25Results = await this.retrieveBM25(query, k);

      // Simple RRF fusion (Reciprocal Rank Fusion)
      const fusedResults = this.rrfFusion(denseResults, bm25Results, k);
      return fusedResults;
    } catch (error) {
      console.error('Hybrid retrieval failed:', error);
      return [];
    }
  }

  private rrfFusion(results1: RetrievalResult[], results2: RetrievalResult[], k: number): RetrievalResult[] {
    const scoreMap = new Map<string, number>();

    results1.forEach((result, index) => {
      const score = 1 / (k + index + 1);
      scoreMap.set(result.chunk.id, (scoreMap.get(result.chunk.id) || 0) + score);
    });

    results2.forEach((result, index) => {
      const score = 1 / (k + index + 1);
      scoreMap.set(result.chunk.id, (scoreMap.get(result.chunk.id) || 0) + score);
    });

    return Array.from(scoreMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([chunkId, score]) => {
        // Find the original chunk from either results
        const original = results1.find(r => r.chunk.id === chunkId) || results2.find(r => r.chunk.id === chunkId);
        return original ? { ...original, score } : null;
      })
      .filter(Boolean) as RetrievalResult[];
  }

  async rerankCrossEncoder(_query: string, candidates: RetrievalResult[], k: number): Promise<RerankResult[]> {
    try {
      // In a real implementation, this would use a cross-encoder model
      // For now, return candidates sorted by score
      const sorted = [...candidates].sort((a, b) => b.score - a.score).slice(0, k);
      return sorted.map((result, index) => ({
        chunk: result.chunk,
        score: result.score * (1 - index * 0.05), // Slightly discount lower ranks
        originalRank: candidates.findIndex(c => c.chunk.id === result.chunk.id) + 1,
        newRank: index + 1,
      }));
    } catch (error) {
      console.error('Cross-encoder reranking failed:', error);
      return candidates.map((result, index) => ({
        chunk: result.chunk,
        score: result.score,
        originalRank: index + 1,
        newRank: index + 1,
      }));
    }
  }

  async rerankLLM(_query: string, candidates: RetrievalResult[], k: number): Promise<RerankResult[]> {
    try {
      // In a real implementation, this would use an LLM to score candidates
      return await this.rerankCrossEncoder(_query, candidates, k);
    } catch (error) {
      console.error('LLM reranking failed:', error);
      return candidates.map((result, index) => ({
        chunk: result.chunk,
        score: result.score,
        originalRank: index + 1,
        newRank: index + 1,
      }));
    }
  }

  async generateAnswer(query: string, chunks: Chunk[], promptTemplate?: string): Promise<GenerationResult> {
    const defaultPrompt = `You are a helpful assistant. Answer the following question based on the provided context.

Question: ${query}

Context:
{context}

Answer:`;
    const prompt = promptTemplate || defaultPrompt;
    const context = chunks.map(c => c.text).join('\n\n');

    return ollamaClient.generateAnswer(this.config.instructModel!, prompt, context);
  }

  calculateMetrics(results: RetrievalResult[], relevantDocuments: string[], k: number): Metrics {
    const topKResults = results.slice(0, k);
    const retrievedDocs = new Set(topKResults.map(r => r.chunk.documentId));
    const relevantSet = new Set(relevantDocuments);

    // Recall@k
    const relevantRetrieved = retrievedDocs.size;
    const relevantTotal = relevantSet.size;
    const recallAtK = relevantTotal > 0 ? relevantRetrieved / relevantTotal : 0;

    // Precision@k
    const precisionAtK = retrievedDocs.size > 0 ? relevantRetrieved / retrievedDocs.size : 0;

    // NDCG@k
    let dcg = 0;
    topKResults.forEach((result, index) => {
      const relevance = relevantSet.has(result.chunk.documentId) ? 1 : 0;
      dcg += relevance / Math.log2(index + 2);
    });

    let idealDcg = 0;
    for (let i = 0; i < Math.min(relevantSet.size, k); i++) {
      idealDcg += 1 / Math.log2(i + 2);
    }
    const ndcgAtK = idealDcg > 0 ? dcg / idealDcg : 0;

    // MRR
    let rr = 0;
    for (let i = 0; i < topKResults.length; i++) {
      if (relevantSet.has(topKResults[i].chunk.documentId)) {
        rr = 1 / (i + 1);
        break;
      }
    }

    return {
      recallAt5: recallAtK,
      precisionAt5: precisionAtK,
      ndcgAt5: ndcgAtK,
      mrr: rr,
    };
  }
}

export const ragService = new RAGService();
