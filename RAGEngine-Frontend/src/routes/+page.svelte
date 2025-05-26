<script lang="ts">
  import { onMount } from 'svelte';
  import { systemInfo, documents, connectionStatus } from '$lib/stores/system';
  import { api } from '$lib/api';

  let loading = true;
  let error: string | null = null;

  onMount(async () => {
    try {
      connectionStatus.set('connecting');
      
      const [sysInfo, docs] = await Promise.all([
        api.getSystemInfo(),
        api.getDocuments()
      ]);
      
      systemInfo.set(sysInfo);
      documents.set(docs);
      connectionStatus.set('connected');
      error = null;
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      connectionStatus.set('error');
      error = err instanceof Error ? err.message : 'Failed to connect to backend';
      
      // Set default values for offline mode
      systemInfo.set({
        collection_info: {
          total_documents: 0,
          total_chunks: 0,
          vector_dimensions: 1536
        }
      });
      documents.set([]);
    } finally {
      loading = false;
    }
  });
</script>

<div class="p-6">
  <div class="mb-8">
    <h2 class="text-3xl font-bold text-white mb-2">Dashboard</h2>
    <p class="text-gray-400">Monitor and control your RAG knowledge base</p>
  </div>

  <!-- Error Message -->
  {#if error}
    <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mb-6">
      <strong>Connection Error:</strong> {error}
      <p class="text-sm mt-1">Make sure the RAGEngine backend is running on http://localhost:8080</p>
    </div>
  {/if}

  <!-- Loading State -->
  {#if loading}
    <div class="bg-gray-800 rounded-lg p-6 mb-8">
      <div class="animate-pulse">
        <div class="h-4 bg-gray-700 rounded w-1/4 mb-4"></div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
          <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
          <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
        </div>
      </div>
    </div>
  {:else}
    <!-- System Overview -->
    <div class="bg-gray-800 rounded-lg p-6 mb-8">
      <h3 class="text-xl font-semibold text-white mb-4">📊 System Overview</h3>
      
      {#if $systemInfo}
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="bg-gray-700 rounded-lg p-4">
            <div class="text-2xl font-bold text-blue-400">{$systemInfo.collection_info?.total_documents || 0}</div>
            <div class="text-gray-300">Documents</div>
          </div>
          
          <div class="bg-gray-700 rounded-lg p-4">
            <div class="text-2xl font-bold text-green-400">{$systemInfo.collection_info?.total_chunks || 0}</div>
            <div class="text-gray-300">Chunks</div>
          </div>
          
          <div class="bg-gray-700 rounded-lg p-4">
            <div class="text-2xl font-bold text-purple-400">{$systemInfo.collection_info?.vector_dimensions || 0}</div>
            <div class="text-gray-300">Vector Dimensions</div>
          </div>
        </div>
      {:else}
        <div class="text-gray-400">System information unavailable</div>
      {/if}
    </div>
  {/if}

  <!-- Quick Search -->
  <div class="bg-gray-800 rounded-lg p-6 mb-8">
    <h3 class="text-xl font-semibold text-white mb-4">🔍 Quick Search</h3>
    
    <form class="space-y-4">
      <div class="flex gap-4">
        <input 
          type="text" 
          placeholder="Enter your search query..." 
          class="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          disabled={error !== null}
        >
        
        <select class="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white" disabled={error !== null}>
          <option value="hybrid">Hybrid Search</option>
          <option value="dense">Dense Only</option>
          <option value="sparse">Sparse Only</option>
        </select>
        
        <button 
          type="submit" 
          class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={error !== null}
        >
          Search
        </button>
      </div>
      
      {#if error}
        <p class="text-red-400 text-sm">Connect to backend to enable search functionality</p>
      {/if}
    </form>
  </div>

  <!-- Recent Documents -->
  <div class="bg-gray-800 rounded-lg p-6">
    <h3 class="text-xl font-semibold text-white mb-4">📄 Recent Documents</h3>
    
    {#if loading}
      <div class="animate-pulse space-y-3">
        <div class="bg-gray-700 rounded-lg p-3 h-16"></div>
        <div class="bg-gray-700 rounded-lg p-3 h-16"></div>
        <div class="bg-gray-700 rounded-lg p-3 h-16"></div>
      </div>
    {:else if $documents && $documents.length > 0}
      <div class="space-y-3">
        {#each $documents.slice(0, 5) as doc}
          <div class="flex items-center justify-between bg-gray-700 rounded-lg p-3">
            <div>
              <div class="text-white font-medium">{doc.filename || doc.title || 'Untitled Document'}</div>
              <div class="text-gray-400 text-sm">{doc.chunks || 0} chunks • {doc.upload_date || 'Unknown date'}</div>
            </div>
            <div class="text-gray-400">
              📄
            </div>
          </div>
        {/each}
      </div>
    {:else}
      <div class="text-gray-400">
        {error ? 'Cannot load documents - backend not connected' : 'No documents found. Upload some documents to get started.'}
      </div>
    {/if}
  </div>
</div>
