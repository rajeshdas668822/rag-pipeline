export interface EvaluationQuery {
  id: string;
  query: string;
  relevantDocuments: string[];
}

export interface ChunkManifest {
  version: string;
  totalChunks: number;
  chunks: ChunkMetadata[];
}

export interface ChunkMetadata {
  id: string;
  documentId: string;
  documentTitle: string;
  textPreview: string;
  embeddingFile: string;
  score?: number;
}
