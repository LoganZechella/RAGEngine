import { browser } from '$app/environment';

// Type definitions
interface RequestOptions extends RequestInit {
  headers?: Record<string, string>;
}

interface SystemInfo {
  collection_info?: {
    total_documents: number;
    total_chunks: number;
    vector_dimensions: number;
  };
  [key: string]: any;
}

interface Document {
  filename?: string;
  title?: string;
  chunks?: number;
  upload_date?: string;
  [key: string]: any;
}

interface SearchResult {
  contexts: any[];
  num_results: number;
  synthesis?: string;
  [key: string]: any;
}

type SearchMode = 'hybrid' | 'dense' | 'sparse';

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
    return this.request('/documents');
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
}

export const api = new RAGEngineAPI(); 