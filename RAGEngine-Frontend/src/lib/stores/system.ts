import { writable } from 'svelte/store';

// Type definitions
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

type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export const systemInfo = writable<SystemInfo | null>(null);
export const documents = writable<Document[]>([]);
export const connectionStatus = writable<ConnectionStatus>('connected'); 