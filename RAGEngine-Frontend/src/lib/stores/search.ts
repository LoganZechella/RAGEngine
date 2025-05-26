import { writable, derived } from 'svelte/store';

// Type definitions
type SearchMode = 'hybrid' | 'dense' | 'sparse';

interface SearchState {
  isSearching: boolean;
  query: string;
  mode: SearchMode;
  synthesize: boolean;
  taskId: string | null;
  results: any | null;
  error: string | null;
}

interface SearchProgress {
  progress: number;
  phase: string;
  message: string;
  currentPhase: number;
  totalPhases: number;
  elapsedTime: number;
  estimatedRemaining: number | null;
  errors: string[];
}

export const searchState = writable<SearchState>({
  isSearching: false,
  query: '',
  mode: 'hybrid',
  synthesize: true,
  taskId: null,
  results: null,
  error: null,
});

export const searchProgress = writable<SearchProgress>({
  progress: 0,
  phase: '',
  message: '',
  currentPhase: 1,
  totalPhases: 8,
  elapsedTime: 0,
  estimatedRemaining: null,
  errors: [],
});

export const searchCompleted = derived(
  searchProgress,
  ($progress) => $progress.progress >= 100
); 