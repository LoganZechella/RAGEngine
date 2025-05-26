import { writable } from 'svelte/store';

// Type definitions matching actual API response structure
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

type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export const systemInfo = writable<SystemInfo | null>(null);
export const documents = writable<Document[]>([]);
export const connectionStatus = writable<ConnectionStatus>('connected'); 