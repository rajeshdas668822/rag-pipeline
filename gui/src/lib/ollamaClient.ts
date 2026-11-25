import type { GenerationResult } from '../types';

const DEFAULT_TIMEOUT = 30000; // 30 seconds

interface OllamaConfig {
  endpoint: string;
  timeout?: number;
}

class OllamaClient {
  private config: OllamaConfig;

  constructor(config: OllamaConfig) {
    this.config = {
      endpoint: config.endpoint || 'http://localhost:11434',
      timeout: config.timeout || DEFAULT_TIMEOUT,
    };
  }

  private async fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
    const { timeout } = this.config;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      return response;
    } finally {
      clearTimeout(id);
    }
  }

  async getModels(): Promise<string[]> {
    try {
      const response = await this.fetchWithTimeout(`${this.config.endpoint}/api/tags`);
      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }
      const data = await response.json();
      return data.models?.map((m: { name: string }) => m.name) || [];
    } catch (error) {
      console.error('Failed to get models:', error);
      throw error instanceof Error
        ? new Error(`Failed to connect to Ollama: ${error.message}`)
        : new Error('Failed to connect to Ollama: Unknown error');
    }
  }

  async generateEmbedding(model: string, text: string): Promise<number[]> {
    try {
      const response = await this.fetchWithTimeout(`${this.config.endpoint}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt: text }),
      });
      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }
      const data = await response.json();
      return data.embedding || [];
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error instanceof Error
        ? new Error(`Failed to generate embedding: ${error.message}`)
        : new Error('Failed to generate embedding: Unknown error');
    }
  }

  async generateAnswer(model: string, prompt: string, context: string): Promise<GenerationResult> {
    const fullPrompt = `${prompt}\n\nContext:\n${context}`;
    try {
      const startTime = Date.now();
      const response = await this.fetchWithTimeout(`${this.config.endpoint}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: fullPrompt,
          stream: false,
        }),
      });
      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }
      const data = await response.json();
      return {
        answer: data.response || '',
        chunks: [],
        metadata: {
          model,
          promptTokens: data.prompt_eval_count || 0,
          completionTokens: data.eval_count || 0,
          latencyMs: Date.now() - startTime,
        },
      };
    } catch (error) {
      console.error('Failed to generate answer:', error);
      throw error instanceof Error
        ? new Error(`Failed to generate answer: ${error.message}`)
        : new Error('Failed to generate answer: Unknown error');
    }
  }
}

export const createOllamaClient = (config: Partial<OllamaConfig> = {}) => {
  return new OllamaClient({
    endpoint: config.endpoint || 'http://localhost:11434',
    timeout: config.timeout || DEFAULT_TIMEOUT,
  });
};

export const ollamaClient = createOllamaClient();
