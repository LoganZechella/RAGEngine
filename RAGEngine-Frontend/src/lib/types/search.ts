export interface SearchProgress {
  progress: number;
  status: 'initializing' | 'processing' | 'completed' | 'error' | 'cancelled';
  phase: string;
  message: string;
  current_phase: number;
  total_phases: number;
  elapsed_time: number;
  estimated_remaining: number | null;
  errors: string[];
  query: string;
}

export interface KnowledgeSynthesis {
  summary: string;
  overall_confidence?: number;
  key_concepts?: KeyConcept[];
  synthesis_insights?: SynthesisInsight[];
  research_gaps?: ResearchGap[];
  topics?: string[];
}

export interface KeyConcept {
  concept: string;
  explanation: string;
  importance?: string;
  evidence_quality?: 'strong' | 'moderate' | 'weak' | 'insufficient' | 'conflicting';
  confidence_score?: number;
}

export interface SynthesisInsight {
  insight: string;
  confidence_level?: 'high' | 'moderate' | 'low';
  implications?: string;
  supporting_evidence?: string[];
}

export interface ResearchGap {
  gap_description: string;
  severity?: 'critical' | 'moderate' | 'minor';
  suggested_investigation?: string;
}

export type SearchMode = 'hybrid' | 'dense' | 'sparse';

export interface SearchState {
  isSearching: boolean;
  query: string;
  mode: SearchMode;
  synthesize: boolean;
  taskId: string | null;
  results: any | null;
  error: string | null;
  synthesis: KnowledgeSynthesis | null;
}

export interface SearchResult {
  contexts: any[];
  num_results: number;
  synthesis?: KnowledgeSynthesis;
  [key: string]: any;
} 