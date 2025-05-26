<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/api';

  type SearchMode = 'hybrid' | 'dense' | 'sparse';

  let query = '';
  let mode: SearchMode = 'hybrid';
  let synthesize = true;
  let searching = false;
  let results: any = null;
  let error: string | null = null;

  async function handleSearch() {
    if (!query.trim()) return;
    
    searching = true;
    error = null;
    
    try {
      // For now, we'll use a simple search call
      // In the future, this could be enhanced with progress tracking
      results = await api.search(query, mode, synthesize);
    } catch (err) {
      console.error('Search failed:', err);
      error = err instanceof Error ? err.message : 'Search failed';
    } finally {
      searching = false;
    }
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
        >
        
        <select bind:value={mode} class="input-field">
          <option value="hybrid">Hybrid Search</option>
          <option value="dense">Dense Only</option>
          <option value="sparse">Sparse Only</option>
        </select>
        
        <label class="flex items-center space-x-2 text-gray-300">
          <input type="checkbox" bind:checked={synthesize} class="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500">
          <span>Synthesize</span>
        </label>
        
        <button 
          type="submit" 
          class="btn-primary flex items-center space-x-2"
          class:searching
          disabled={searching}
        >
          <span class="search-icon">{searching ? '⏳' : '🔍'}</span>
          <span>{searching ? 'Searching...' : 'Search'}</span>
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

  <!-- Error Display -->
  {#if error}
    <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mb-6">
      <strong>Search Error:</strong> {error}
    </div>
  {/if}

  <!-- Search Results -->
  {#if results}
    <div class="card">
      <h3 class="text-xl font-semibold text-white mb-4">Search Results</h3>
      
      {#if results.synthesis}
        <div class="bg-blue-900 border border-blue-700 rounded-lg p-4 mb-6">
          <h4 class="text-lg font-semibold text-blue-200 mb-2">🧠 AI Synthesis</h4>
          <p class="text-blue-100">{results.synthesis}</p>
        </div>
      {/if}

      {#if results.contexts && results.contexts.length > 0}
        <div class="space-y-4">
          <h4 class="text-lg font-semibold text-white">Found {results.contexts.length} relevant documents:</h4>
          
          {#each results.contexts as context, i}
            <div class="bg-gray-700 rounded-lg p-4">
              <div class="flex justify-between items-start mb-2">
                <h5 class="font-semibold text-white">{context.metadata?.filename || `Document ${i + 1}`}</h5>
                <span class="text-sm text-gray-400">Score: {(context.score || 0).toFixed(3)}</span>
              </div>
              <p class="text-gray-300 text-sm mb-2">{context.content}</p>
              {#if context.metadata}
                <div class="text-xs text-gray-500">
                  {#if context.metadata.page}Page {context.metadata.page}{/if}
                  {#if context.metadata.chunk_id} • Chunk {context.metadata.chunk_id}{/if}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="text-gray-400">No results found for your query.</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .search-icon {
    display: inline-block;
    transition: transform 0.3s ease;
  }
  
  .searching .search-icon {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style> 