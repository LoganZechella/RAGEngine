<script lang="ts">
  import { onMount } from 'svelte';
  import { systemInfo, connectionStatus, documents } from '$lib/stores/system';
  import { api } from '$lib/api';

  let loading = false;
  let message: string | null = null;
  let messageType: 'success' | 'error' | 'info' = 'info';

  async function handleCollectionAction(action: string) {
    loading = true;
    message = null;
    
    try {
      let result;
      switch (action) {
        case 'clear':
          result = await api.clearCollection();
          message = 'Collection cleared successfully';
          messageType = 'success';
          break;
        case 'recreate':
          result = await api.recreateCollection();
          message = 'Collection recreated successfully';
          messageType = 'success';
          break;
        default:
          throw new Error('Unknown action');
      }
    } catch (err) {
      console.error(`Collection ${action} failed:`, err);
      message = err instanceof Error ? err.message : `Failed to ${action} collection`;
      messageType = 'error';
    } finally {
      loading = false;
    }
  }

  async function refreshSystemInfo() {
    loading = true;
    message = null;
    
    try {
      const info = await api.getSystemInfo();
      systemInfo.set(info);
      message = 'System information refreshed';
      messageType = 'success';
    } catch (err) {
      console.error('Failed to refresh system info:', err);
      message = err instanceof Error ? err.message : 'Failed to refresh system info';
      messageType = 'error';
    } finally {
      loading = false;
    }
  }

  function clearMessage() {
    message = null;
  }

  // Load initial data when component mounts
  onMount(async () => {
    try {
      // Load system info
      const info = await api.getSystemInfo();
      systemInfo.set(info);
      
      // Load documents
      const docs = await api.getDocuments();
      documents.set(docs);
      
      // Set connection status to connected
      connectionStatus.set('connected');
    } catch (err) {
      console.error('Failed to load initial data:', err);
      connectionStatus.set('error');
    }
  });
</script>

<div class="p-6">
  <div class="mb-8">
    <h2 class="text-3xl font-bold text-white mb-2">Settings</h2>
    <p class="text-gray-400">Configure and manage your RAGEngine instance</p>
  </div>

  <!-- Status Message -->
  {#if message}
    <div class="mb-6 px-4 py-3 rounded border {messageType === 'success' ? 'bg-green-900 border-green-700 text-green-200' : messageType === 'error' ? 'bg-red-900 border-red-700 text-red-200' : 'bg-blue-900 border-blue-700 text-blue-200'}">
      <div class="flex justify-between items-center">
        <span>{message}</span>
        <button on:click={clearMessage} class="text-xl hover:opacity-70">×</button>
      </div>
    </div>
  {/if}

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- System Information -->
    <div class="card">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-xl font-semibold text-white">⚙️ System Information</h3>
        <button 
          on:click={refreshSystemInfo}
          class="btn-secondary"
          disabled={loading}
        >
          {loading ? '⏳' : '🔄'} Refresh
        </button>
      </div>
      
      {#if $systemInfo}
        <div class="space-y-4">
          <div class="bg-gray-700 rounded-lg p-4">
            <h4 class="font-semibold text-white mb-2">Collection Configuration</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Collection Name:</span>
                <span class="text-white">{$systemInfo.config?.collection_name || $systemInfo.vector_db?.name || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Vector Dimensions:</span>
                <span class="text-white">{$systemInfo.config?.vector_dimensions || $systemInfo.vector_db?.vector_size || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Chunk Size:</span>
                <span class="text-white">{$systemInfo.ingestion?.max_chunk_size_tokens || 'Unknown'} tokens</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Chunk Overlap:</span>
                <span class="text-white">{$systemInfo.ingestion?.chunk_overlap_tokens || 'Unknown'} tokens</span>
              </div>
            </div>
          </div>

          <div class="bg-gray-700 rounded-lg p-4">
            <h4 class="font-semibold text-white mb-2">Collection Statistics</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Total Documents:</span>
                <span class="text-white">{$documents?.length || 0}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Total Chunks:</span>
                <span class="text-white">{$systemInfo.vector_db?.points_count || 0}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Index Size:</span>
                <span class="text-white">{$systemInfo.vector_db?.segments_count ? `${$systemInfo.vector_db.segments_count} segments` : 'Unknown'}</span>
              </div>
            </div>
          </div>

          <div class="bg-gray-700 rounded-lg p-4">
            <h4 class="font-semibold text-white mb-2">Search Configuration</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Top K Dense:</span>
                <span class="text-white">{$systemInfo.rag_engine?.hybrid_searcher?.top_k_dense || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Top K Sparse:</span>
                <span class="text-white">{$systemInfo.rag_engine?.hybrid_searcher?.top_k_sparse || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Top K Rerank:</span>
                <span class="text-white">{$systemInfo.rag_engine?.top_k_rerank || 'Unknown'}</span>
              </div>
            </div>
          </div>
        </div>
      {:else}
        <div class="text-gray-400">System information not available</div>
      {/if}
    </div>

    <!-- Collection Management -->
    <div class="card">
      <h3 class="text-xl font-semibold text-white mb-4">🗄️ Collection Management</h3>
      
      <div class="space-y-4">
        <div class="bg-gray-700 rounded-lg p-4">
          <h4 class="font-semibold text-white mb-2">Clear Collection</h4>
          <p class="text-gray-400 text-sm mb-3">Remove all documents and vectors from the collection while keeping the collection structure.</p>
          <button 
            on:click={() => {
              if (confirm('Are you sure you want to clear all data from the collection?')) {
                handleCollectionAction('clear');
              }
            }}
            class="w-full bg-yellow-600 hover:bg-yellow-700 text-white py-2 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? '⏳ Processing...' : '🗑️ Clear Collection'}
          </button>
        </div>

        <div class="bg-gray-700 rounded-lg p-4">
          <h4 class="font-semibold text-white mb-2">Recreate Collection</h4>
          <p class="text-gray-400 text-sm mb-3">Delete and recreate the entire collection. This will remove all data and reset the collection to its initial state.</p>
          <button 
            on:click={() => {
              if (confirm('Are you sure you want to recreate the collection? This will delete all data.')) {
                handleCollectionAction('recreate');
              }
            }}
            class="w-full bg-red-600 hover:bg-red-700 text-white py-2 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? '⏳ Processing...' : '🔄 Recreate Collection'}
          </button>
        </div>
      </div>
    </div>

    <!-- Connection Status -->
    <div class="card">
      <h3 class="text-xl font-semibold text-white mb-4">🔗 Connection Status</h3>
      
      <div class="space-y-4">
        <div class="flex items-center space-x-3">
          <div class="w-4 h-4 rounded-full {$connectionStatus === 'connected' ? 'bg-green-500' : $connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'}"></div>
          <span class="text-white font-medium">
            {$connectionStatus === 'connected' ? 'Connected' : $connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </span>
        </div>
        
        <div class="bg-gray-700 rounded-lg p-4">
          <h4 class="font-semibold text-white mb-2">Backend Information</h4>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-400">API Endpoint:</span>
              <span class="text-white">http://localhost:8080</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Status:</span>
              <span class="text-white capitalize">{$connectionStatus}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Environment Information -->
    <div class="card">
      <h3 class="text-xl font-semibold text-white mb-4">🌍 Environment</h3>
      
      <div class="bg-gray-700 rounded-lg p-4">
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Frontend:</span>
            <span class="text-white">SvelteKit</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Backend:</span>
            <span class="text-white">FastAPI + RAGEngine</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Vector Database:</span>
            <span class="text-white">Qdrant</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Embeddings:</span>
            <span class="text-white">OpenAI</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div> 