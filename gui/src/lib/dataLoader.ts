import type { EvaluationQuery, ChunkManifest, ChunkMetadata } from '../types';

interface DataLoaderConfig {
  dataDir?: string;
}

class DataLoader {
  private config: DataLoaderConfig;

  constructor(config: DataLoaderConfig = {}) {
    this.config = {
      dataDir: config.dataDir || '../data',
    };
  }

  async loadEvaluationQueries(): Promise<EvaluationQuery[]> {
    try {
      const response = await fetch(`${this.config.dataDir}/evaluation_queries.jsonl`);
      if (!response.ok) {
        throw new Error(`Failed to load evaluation queries: ${response.statusText}`);
      }
      const data = await response.text();
      return data
        .split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line));
    } catch (error) {
      console.error('Failed to load evaluation queries:', error);
      return [];
    }
  }

  async loadEvaluationLabels(): Promise<Record<string, string[]>> {
    try {
      const queries = await this.loadEvaluationQueries();
      return Object.fromEntries(queries.map(q => [q.id, q.relevantDocuments || []]));
    } catch (error) {
      console.error('Failed to load evaluation labels:', error);
      return {};
    }
  }

  async loadChunksManifest(): Promise<ChunkManifest> {
    try {
      const response = await fetch(`${this.config.dataDir}/chunks_manifest.json`);
      if (!response.ok) {
        throw new Error(`Failed to load chunks manifest: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to load chunks manifest:', error);
      return {
        version: '1.0',
        totalChunks: 0,
        chunks: [],
      };
    }
  }

  async loadChunkText(chunkId: string): Promise<string> {
    try {
      // In a real implementation, this would load from the chunk storage
      // For now, return a placeholder
      return `Chunk text for ${chunkId} - this would be loaded from storage in a real implementation.`;
    } catch (error) {
      console.error(`Failed to load chunk text for ${chunkId}:`, error);
      return `Error loading chunk: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }

  async preloadChunks(chunks: ChunkMetadata[]): Promise<Map<string, string>> {
    const chunkMap = new Map<string, string>();
    for (const chunk of chunks) {
      try {
        const text = await this.loadChunkText(chunk.id);
        chunkMap.set(chunk.id, text);
      } catch (error) {
        console.error(`Failed to preload chunk ${chunk.id}:`, error);
      }
    }
    return chunkMap;
  }
}

export const dataLoader = new DataLoader();
