import { create } from 'zustand';
import type { RetrievalStrategy, RerankerType, RAGConfig } from '../types';

interface ConfigState {
  config: RAGConfig;
  models: string[];
  loadingModels: boolean;
  error: string | null;

  // Actions
  setRetrievalStrategy: (strategy: RetrievalStrategy) => void;
  setRerankingEnabled: (enabled: boolean) => void;
  setRerankerType: (type: RerankerType) => void;
  setTopK: (k: number) => void;
  setOllamaEndpoint: (endpoint: string) => void;
  setEmbeddingModel: (model: string) => void;
  setInstructModel: (model: string) => void;
  setModels: (models: string[]) => void;
  setLoadingModels: (loading: boolean) => void;
  setError: (error: string | null) => void;
  loadModels: () => Promise<void>;
  reset: () => void;
}

const STORAGE_KEY = 'rag-gui-config';

const loadFromStorage = (): Partial<RAGConfig> => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Failed to load config from storage:', error);
  }
  return {};
};

export const useConfigStore = create<ConfigState>((set, get) => ({
  config: {
    retrievalStrategy: 'hybrid',
    rerankingEnabled: true,
    rerankerType: 'cross-encoder',
    topK: 10,
    ollamaEndpoint: 'http://localhost:11434',
    embeddingModel: 'nomic-embed-text',
    instructModel: 'mistral',
    ...loadFromStorage(),
  },
  models: [],
  loadingModels: false,
  error: null,

  setRetrievalStrategy: strategy =>
    set(state => {
      const config = { ...state.config, retrievalStrategy: strategy };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setRerankingEnabled: enabled =>
    set(state => {
      const config = { ...state.config, rerankingEnabled: enabled };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setRerankerType: type =>
    set(state => {
      const config = { ...state.config, rerankerType: type };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setTopK: k =>
    set(state => {
      const config = { ...state.config, topK: k };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setOllamaEndpoint: endpoint =>
    set(state => {
      const config = { ...state.config, ollamaEndpoint: endpoint };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setEmbeddingModel: model =>
    set(state => {
      const config = { ...state.config, embeddingModel: model };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setInstructModel: model =>
    set(state => {
      const config = { ...state.config, instructModel: model };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      return { config };
    }),

  setModels: models => set({ models }),

  setLoadingModels: loading => set({ loadingModels: loading }),

  setError: error => set({ error }),

  loadModels: async () => {
    const { setModels, setLoadingModels, setError } = get();
    setLoadingModels(true);
    setError(null);
    try {
      const response = await fetch(`${get().config.ollamaEndpoint}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        const modelNames = data.models?.map((m: { name: string }) => m.name) || [];
        setModels(modelNames);
      } else {
        setError('Failed to fetch models from Ollama');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to connect to Ollama');
    } finally {
      setLoadingModels(false);
    }
  },

  reset: () => {
    localStorage.removeItem(STORAGE_KEY);
    set({
      config: {
        retrievalStrategy: 'hybrid',
        rerankingEnabled: true,
        rerankerType: 'cross-encoder',
        topK: 10,
        ollamaEndpoint: 'http://localhost:11434',
        embeddingModel: 'nomic-embed-text',
        instructModel: 'mistral',
      },
      error: null,
    });
  },
}));
