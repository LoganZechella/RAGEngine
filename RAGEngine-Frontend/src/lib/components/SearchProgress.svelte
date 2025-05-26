<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { searchProgress, searchState, searchPhases } from '$lib/stores/search';
  import { api } from '$lib/api';

  export let taskId: string;
  export let onComplete: () => void;
  export let onCancel: () => void;

  let eventSource: EventSource | null = null;
  let startTime = Date.now();
  let connectionStatus = 'disconnected';
  let lastMessageTime: number | null = null;
  let messageCount = 0;
  let debugLogs: string[] = [];

  function addDebugLog(message: string) {
    const timestamp = new Date().toLocaleTimeString();
    debugLogs = [`[${timestamp}] ${message}`, ...debugLogs.slice(0, 9)]; // Keep last 10 logs
    console.log(`[SearchProgress] ${message}`);
  }

  onMount(() => {
    addDebugLog(`Component mounted with taskId: ${taskId}`);
    connectToProgress();
  });

  onDestroy(() => {
    addDebugLog('Component destroying, closing SSE connection');
    if (eventSource) {
      eventSource.close();
    }
  });

  function connectToProgress() {
    // Validate task ID before attempting connection
    if (!taskId || taskId === 'null' || taskId === 'undefined') {
      addDebugLog(`ERROR: Invalid task ID: ${taskId}`);
      connectionStatus = 'error';
      return;
    }
    
    addDebugLog('Attempting to create SSE connection...');
    
    try {
      eventSource = api.createSearchProgressSSE(taskId);
      
      if (!eventSource) {
        addDebugLog('ERROR: Failed to create EventSource (browser not supported?)');
        connectionStatus = 'error';
        return;
      }

      addDebugLog(`SSE connection created for URL: ${eventSource.url}`);
      connectionStatus = 'connecting';

      eventSource.onopen = (event) => {
        addDebugLog('SSE connection opened successfully');
        connectionStatus = 'connected';
      };

      eventSource.onmessage = (event) => {
        messageCount++;
        lastMessageTime = Date.now();
        addDebugLog(`Received SSE message #${messageCount}: ${event.data.substring(0, 100)}...`);
        
        try {
          const data = JSON.parse(event.data);
          
          // Handle connection confirmation
          if (data.status === 'connected') {
            addDebugLog('SSE connection confirmed by server');
            connectionStatus = 'connected';
            return;
          }
          
          // Handle error messages
          if (data.error) {
            addDebugLog(`Server error: ${data.error}`);
            connectionStatus = 'error';
            return;
          }
          
          // Handle progress updates
          if (data.progress !== undefined) {
            addDebugLog(`Parsed data - Progress: ${data.progress}%, Phase: ${data.phase}, Status: ${data.status}`);
            
            // Update the search progress store
            searchProgress.set(data);
            
            if (data.status === 'completed') {
              addDebugLog('Search completed, calling onComplete callback');
              onComplete();
            }
          }
        } catch (error) {
          addDebugLog(`ERROR parsing SSE data: ${error}`);
          console.error('Error parsing SSE data:', error, 'Raw data:', event.data);
        }
      };

      eventSource.onerror = (error) => {
        addDebugLog(`SSE connection error: ${error}`);
        console.error('SSE connection error:', error);
        connectionStatus = 'error';
        
        // Try to reconnect after a delay
        setTimeout(() => {
          if (connectionStatus === 'error') {
            addDebugLog('Attempting to reconnect...');
            connectToProgress();
          }
        }, 3000);
      };

      // Set up a timeout to detect if no messages are received
      setTimeout(() => {
        if (messageCount === 0) {
          addDebugLog('WARNING: No SSE messages received after 5 seconds');
        }
      }, 5000);

    } catch (error) {
      addDebugLog(`ERROR creating SSE connection: ${error}`);
      console.error('Error creating SSE connection:', error);
      connectionStatus = 'error';
    }
  }

  async function handleCancel() {
    if (confirm('Are you sure you want to cancel this search?')) {
      try {
        addDebugLog('Cancelling search...');
        await api.cancelSearch(taskId);
        onCancel();
      } catch (error) {
        addDebugLog(`ERROR cancelling search: ${error}`);
        console.error('Failed to cancel search:', error);
      }
    }
  }

  // Reactive time calculations
  $: elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
  $: remainingTime = $searchProgress.estimated_remaining ? 
    Math.max(0, Math.round($searchProgress.estimated_remaining)) : null;
  $: timeSinceLastMessage = lastMessageTime ? Math.floor((Date.now() - lastMessageTime) / 1000) : null;
</script>

<div class="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6">
  <!-- Debug Panel (only show in development) -->
  {#if import.meta.env.DEV}
    <div class="bg-black bg-opacity-50 rounded-lg p-3 mb-4 text-xs">
      <div class="flex items-center justify-between mb-2">
        <h5 class="text-yellow-400 font-semibold">Debug Info</h5>
        <div class="flex items-center space-x-2">
          <span class="text-gray-400">Connection:</span>
          <span class="px-2 py-1 rounded text-xs" class:bg-green-600={connectionStatus === 'connected'} 
                class:bg-yellow-600={connectionStatus === 'connecting'} 
                class:bg-red-600={connectionStatus === 'error'}>
            {connectionStatus}
          </span>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-2 text-gray-300 mb-2">
        <div>Task ID: {taskId}</div>
        <div>Messages: {messageCount}</div>
        <div>Last message: {timeSinceLastMessage ? `${timeSinceLastMessage}s ago` : 'None'}</div>
        <div>Progress: {$searchProgress.progress}%</div>
      </div>
      <div class="max-h-20 overflow-y-auto">
        {#each debugLogs as log}
          <div class="text-gray-400 text-xs">{log}</div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Header -->
  <div class="flex items-center justify-between mb-6">
    <div>
      <h4 class="text-xl font-semibold text-white mb-1">🔍 Searching Knowledge Base</h4>
      <p class="text-blue-200 text-sm">
        Query: "<span class="font-medium">{$searchProgress.query || 'Loading...'}</span>" 
        <span class="ml-2 px-2 py-1 bg-blue-700 rounded text-xs">hybrid</span>
        <span class="ml-1 px-2 py-1 bg-purple-700 rounded text-xs">AI Analysis</span>
      </p>
    </div>
    <button 
      on:click={handleCancel}
      class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
    >
      Cancel
    </button>
  </div>

  <!-- Progress Bar -->
  <div class="mb-6">
    <div class="flex items-center justify-between mb-2">
      <span class="text-white font-medium">{$searchProgress.phase || 'Initializing...'}</span>
      <div class="flex items-center space-x-3 text-sm text-blue-200">
        <span>{$searchProgress.progress}%</span>
        <span>{elapsedSeconds}s elapsed</span>
        {#if remainingTime}
          <span class="text-blue-300">~{remainingTime}s remaining</span>
        {/if}
      </div>
    </div>
    <div class="w-full bg-gray-700 rounded-full h-3 relative overflow-hidden">
      <div 
        class="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out" 
        style="width: {$searchProgress.progress}%"
      ></div>
    </div>
  </div>

  <!-- Phase Indicators -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
    {#each $searchPhases as phase, i}
      <div class="bg-gray-800 rounded-lg p-3 transition-all duration-300" 
           class:!bg-green-800={phase.completed} 
           class:!bg-blue-800={i + 1 === $searchProgress.current_phase}>
        <div class="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center text-xs mr-2"
             class:!bg-green-500={phase.completed}
             class:!bg-blue-500={i + 1 === $searchProgress.current_phase}>
          {#if phase.completed}
            ✓
          {:else}
            {i + 1}
          {/if}
        </div>
        <span>{phase.name}</span>
      </div>
    {/each}
  </div>

  <!-- Status Messages -->
  <div class="bg-black bg-opacity-30 rounded-lg p-4">
    <div class="flex items-start space-x-3">
      <div class="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      <div>
        <p class="text-white font-medium">{$searchProgress.message || 'Connecting to search service...'}</p>
        <div class="text-blue-200 text-sm">
          <p>Progress: {$searchProgress.progress}% (Phase {$searchProgress.current_phase}/{$searchProgress.total_phases})</p>
          {#if connectionStatus !== 'connected'}
            <p class="text-yellow-300">⚠️ Connection status: {connectionStatus}</p>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

 