import { browser } from '$app/environment';
import type { SearchMode, SearchResult } from './types/search';

// Type definitions matching actual API response structure
interface RequestOptions extends RequestInit {
  headers?: Record<string, string>;
}

interface SystemInfo {
  config?: {
    collection_name?: string;
    chunking_strategy?: string;
    vector_dimensions?: number;
    source_paths?: string[];
  };
  ingestion?: {
    source_paths?: string[];
    chunking_strategy?: string;
    max_chunk_size_tokens?: number;
    chunk_overlap_tokens?: number;
    vector_dimensions?: number;
  };
  rag_engine?: {
    hybrid_searcher?: {
      top_k_dense?: number;
      top_k_sparse?: number;
      rrf_k?: number;
      bm25_index_available?: boolean;
      bm25_corpus_size?: number;
      vector_db_available?: boolean;
      embedding_generator_available?: boolean;
    };
    reranker?: {
      model?: string;
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      api_available?: boolean;
      rate_limit_interval?: number;
      context_window?: string;
    };
    deep_analyzer?: {
      model?: string;
      max_tokens?: number;
      temperature?: number;
      api_available?: boolean;
      analysis_capabilities?: string[];
    };
    top_k_rerank?: number;
  };
  vector_db?: {
    name?: string;
    vector_size?: number;
    distance?: string;
    points_count?: number;
    vectors_count?: number | null;
    indexed_vectors_count?: number;
    payload_schema?: Record<string, any>;
    status?: string;
    segments_count?: number;
  };
  [key: string]: any;
}

interface DocumentInfo {
  size: number;
  mtime: number;
  hash: string;
  last_processed: string;
  processing_success: boolean;
}

interface Document {
  filename: string;
  title?: string;
  size?: number;
  chunks?: number;
  upload_date?: string;
  status?: string;
  error_message?: string;
  file_size?: number;
  [key: string]: any;
}



class RAGEngineAPI {
  private baseURL: string;

  constructor() {
    this.baseURL = browser ? 
      (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080') : 
      'http://localhost:8080';
  }

  async request(endpoint: string, options: RequestOptions = {}): Promise<any> {
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Search methods
  async search(query: string, mode: SearchMode = 'hybrid', synthesize: boolean = true): Promise<SearchResult> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('mode', mode);
    formData.append('synthesize', synthesize.toString());

    try {
      const response = await fetch(`${this.baseURL}/api/search`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  }

  async startSearch(query: string, mode: SearchMode = 'hybrid', synthesize: boolean = true): Promise<string> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('mode', mode);
    formData.append('synthesize', synthesize.toString());

    return fetch(`${this.baseURL}/search-with-progress`, {
      method: 'POST',
      body: formData,
    }).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.text(); // Return HTML for now, will be JSON later
    });
  }

  createSSEConnection(endpoint: string): EventSource | null {
    if (!browser) return null;
    return new EventSource(`${this.baseURL}${endpoint}`);
  }

  // System methods
  async getSystemInfo(): Promise<SystemInfo> {
    return this.request('/system-info');
  }

  async getDocuments(): Promise<Document[]> {
    const response = await this.request('/documents');
    
    // Transform the API response (dictionary with paths as keys) into an array
    const documents: Document[] = [];
    for (const [fullPath, info] of Object.entries(response as Record<string, DocumentInfo>)) {
      const filename = fullPath.split('/').pop() || fullPath;
      documents.push({
        filename,
        title: filename,
        size: info.size,
        file_size: info.size,
        upload_date: info.last_processed,
        status: info.processing_success ? 'processed' : 'error',
        chunks: undefined, // This info isn't available in the current API response
        error_message: info.processing_success ? undefined : 'Processing failed'
      });
    }
    
    return documents;
  }

  // Collection management
  async clearCollection(): Promise<any> {
    return this.request('/collection/clear', { method: 'POST' });
  }

  async recreateCollection(): Promise<any> {
    return this.request('/collection/recreate', { method: 'POST' });
  }

  // File upload
  async uploadFiles(formData: FormData): Promise<string> {
    return fetch(`${this.baseURL}/upload`, {
      method: 'POST',
      body: formData,
    }).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.text(); // Return HTML for now, will be JSON later
    });
  }

  // Enhanced search methods for progress tracking
  async startSearchWithProgress(query: string, mode: SearchMode = 'hybrid', synthesize: boolean = true): Promise<string> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('mode', mode);
    formData.append('synthesize', synthesize.toString());

    const response = await fetch(`${this.baseURL}/api/search-with-progress`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    if (!result.task_id) {
      throw new Error('No task ID returned from server');
    }
    
    return result.task_id;
  }

  createSearchProgressSSE(taskId: string): EventSource | null {
    if (!browser) return null;
    return new EventSource(`${this.baseURL}/search-progress/${taskId}`);
  }

  async getSearchResults(taskId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/search-results/${taskId}`);
    if (!response.ok) {
      if (response.status === 202) {
        throw new Error('Results not ready yet');
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  async cancelSearch(taskId: string): Promise<void> {
    await fetch(`${this.baseURL}/cancel-search/${taskId}`, { method: 'POST' });
  }
}

export const api = new RAGEngineAPI(); 