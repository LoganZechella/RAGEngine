<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { get } from 'svelte/store';
  import { searchState, searchProgress } from '$lib/stores/search';
  import { api } from '$lib/api';
  import SearchProgress from '$lib/components/SearchProgress.svelte';
  import KnowledgeSynthesis from '$lib/components/KnowledgeSynthesis.svelte';
  import { formatScoreBar } from '$lib/utils/formatting';
  import type { SearchMode } from '$lib/types/search';

  let query = '';
  let mode: SearchMode = 'hybrid';
  let synthesize = true;
  let showProgress = false;
  let results: any = null;
  let error: string | null = null;

  async function handleSearch() {
    if (!query.trim()) return;
    
    try {
      error = null;
      showProgress = true;
      
      // Start search with progress
      const taskId = await api.startSearchWithProgress(query, mode, synthesize);
      
      searchState.update(state => ({
        ...state,
        isSearching: true,
        query,
        mode,
        synthesize,
        taskId,
        results: null,
        error: null
      }));
      
    } catch (err) {
      console.error('Search failed:', err);
      error = err instanceof Error ? err.message : 'Search failed';
      showProgress = false;
    }
  }

  async function handleSearchComplete() {
    const currentState = get(searchState);
    if (!currentState.taskId) return;

    try {
      // Get final results
      const response = await api.getSearchResults(currentState.taskId);
      
      // Extract results from JSON response
      results = response.results;
      
      searchState.update(state => ({
        ...state,
        isSearching: false,
        results: response.results,
        synthesis: response.results?.synthesis || null
      }));
      
      showProgress = false;
    } catch (err) {
      console.error('Failed to get results:', err);
      error = 'Failed to retrieve search results';
      showProgress = false;
    }
  }

  function handleSearchCancel() {
    searchState.update(state => ({
      ...state,
      isSearching: false,
      taskId: null
    }));
    
    showProgress = false;
    error = 'Search cancelled';
  }

  function quickSearch(searchQuery: string) {
    query = searchQuery;
    handleSearch();
  }

  // Handle URL parameters on mount
  onMount(() => {
    const urlParams = $page.url.searchParams;
    const urlQuery = urlParams.get('q');
    const urlMode = urlParams.get('mode') as SearchMode;
    
    if (urlQuery) {
      query = urlQuery;
      if (urlMode && ['hybrid', 'dense', 'sparse'].includes(urlMode)) {
        mode = urlMode;
      }
      // Auto-search if query is provided via URL
      handleSearch();
    }
  });
</script>

<div class="p-6">
  <div class="mb-8">
    <h2 class="text-3xl font-bold text-white mb-2">Search</h2>
    <p class="text-gray-400">Search your knowledge base with advanced AI-powered retrieval</p>
  </div>

  <!-- Search Interface -->
  <div class="card mb-8">
    <h3 class="text-xl font-semibold text-white mb-4">🔍 Search Knowledge Base</h3>
    
    <form on:submit|preventDefault={handleSearch} class="space-y-4">
      <div class="flex gap-4">
        <input 
          type="text" 
          bind:value={query}
          placeholder="Enter your search query..." 
          class="flex-1 input-field"
          required
          disabled={showProgress}
        >
        
        <select bind:value={mode} class="input-field" disabled={showProgress}>
          <option value="hybrid">Hybrid Search</option>
          <option value="dense">Dense Only</option>
          <option value="sparse">Sparse Only</option>
        </select>
        
        <label class="flex items-center space-x-2 text-gray-300">
          <input 
            type="checkbox" 
            bind:checked={synthesize} 
            class="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
            disabled={showProgress}
          >
          <span>Synthesize</span>
        </label>
        
        <button 
          type="submit" 
          class="btn-primary flex items-center space-x-2"
          disabled={showProgress}
        >
          <span>{showProgress ? '⏳' : '🔍'}</span>
          <span>{showProgress ? 'Searching...' : 'Search'}</span>
        </button>
      </div>
      
      <!-- Quick search suggestions -->
      <div class="flex flex-wrap gap-2 text-sm">
        <span class="text-gray-400">Quick searches:</span>
        <button type="button" class="text-blue-400 hover:text-blue-300 underline" on:click={() => quickSearch('machine learning algorithms')}>Machine Learning</button>
        <button type="button" class="text-blue-400 hover:text-blue-300 underline" on:click={() => quickSearch('neural networks')}>Neural Networks</button>
        <button type="button" class="text-blue-400 hover:text-blue-300 underline" on:click={() => quickSearch('data processing')}>Data Processing</button>
      </div>
    </form>
  </div>

  <!-- Search Progress -->
  {#if showProgress && $searchState.taskId && $searchState.taskId !== 'null'}
    <SearchProgress 
      taskId={$searchState.taskId}
      onComplete={handleSearchComplete}
      onCancel={handleSearchCancel}
    />
  {:else if showProgress}
    <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mb-6">
      <strong>Search Error:</strong> Failed to start search - invalid task ID
    </div>
  {/if}

  <!-- Error Display -->
  {#if error}
    <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mb-6">
      <strong>Search Error:</strong> {error}
    </div>
  {/if}

  <!-- Search Results -->
  {#if results && !showProgress}
    <div class="space-y-6">
      <!-- Query Info -->
      <div class="bg-gray-700 rounded-lg p-4">
        <p class="text-gray-300">
          <strong>Query:</strong> "{$searchState.query}" 
          <span class="ml-2 px-2 py-1 bg-blue-600 rounded text-sm">{$searchState.mode}</span>
        </p>
        <p class="text-gray-400 text-sm">Found {results.num_results || 0} results</p>
      </div>

      <!-- Knowledge Synthesis -->
      {#if results.synthesis}
        <KnowledgeSynthesis synthesis={results.synthesis} />
      {/if}

      <!-- Retrieved Contexts -->
      {#if results.contexts && results.contexts.length > 0}
        <div class="card">
          <h3 class="text-xl font-semibold text-white mb-4">📄 Retrieved Contexts ({results.contexts.length})</h3>
          <div class="space-y-4">
            {#each results.contexts.slice(0, 5) as context, i}
              <div class="bg-gray-700 rounded-lg p-4">
                <div class="flex items-start justify-between mb-3">
                  <div class="flex-1">
                    <div class="flex items-center space-x-3 mb-2">
                      <h4 class="font-medium text-white">
                        {context.metadata?.document_id || `Document ${i + 1}`}
                      </h4>
                      {#if context.rerank_score}
                        <div class="flex items-center space-x-2 text-sm">
                          <span class="text-gray-400">Rerank:</span>
                          <span class="font-mono text-green-400">{formatScoreBar(context.rerank_score)}</span>
                          <span class="text-gray-400">{Math.round(context.rerank_score * 100)}%</span>
                        </div>
                      {/if}
                      {#if context.initial_score}
                        <div class="flex items-center space-x-2 text-sm">
                          <span class="text-gray-400">Initial:</span>
                          <span class="font-mono text-green-400">{formatScoreBar(context.initial_score)}</span>
                          <span class="text-gray-400">{Math.round(context.initial_score * 100)}%</span>
                        </div>
                      {/if}
                    </div>
                    <p class="text-gray-300 text-sm leading-relaxed">{context.text}</p>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

 