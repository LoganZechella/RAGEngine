import { writable, derived } from 'svelte/store';
import type { SearchProgress, SearchState, KnowledgeSynthesis } from '../types/search';

export const searchState = writable<SearchState>({
  isSearching: false,
  query: '',
  mode: 'hybrid',
  synthesize: true,
  taskId: null,
  results: null,
  error: null,
  synthesis: null,
});

export const searchProgress = writable<SearchProgress>({
  progress: 0,
  status: 'initializing',
  phase: '',
  message: '',
  current_phase: 1,
  total_phases: 8,
  elapsed_time: 0,
  estimated_remaining: null,
  errors: [],
  query: '',
});

// Derived stores
export const searchCompleted = derived(
  searchProgress,
  ($progress) => $progress.status === 'completed'
);

export const searchPhases = derived(
  searchProgress,
  ($progress) => {
    const phases = [
      { name: 'Initialize', completed: $progress.current_phase > 1 },
      { name: 'Embedding', completed: $progress.current_phase > 2 },
      { name: 'Vector Search', completed: $progress.current_phase > 3 },
      { name: 'Analysis', completed: $progress.current_phase > 7 },
    ];
    
    // Add conditional phases based on mode
    if ($progress.total_phases > 6) {
      phases.splice(3, 0, 
        { name: 'Keyword Search', completed: $progress.current_phase > 4 },
        { name: 'Result Fusion', completed: $progress.current_phase > 5 },
        { name: 'Re-ranking', completed: $progress.current_phase > 6 }
      );
    }
    
    return phases;
  }
); 