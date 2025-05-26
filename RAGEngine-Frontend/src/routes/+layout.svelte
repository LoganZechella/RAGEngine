<script lang="ts">
  import '../app.css';
  import { page } from '$app/stores';
  import { connectionStatus } from '$lib/stores/system';
  import { browser } from '$app/environment';
  
  // Reactive statement to safely get pathname
  $: currentPath = browser && $page && $page.url ? $page.url.pathname : '/';
</script>

<div class="flex h-screen bg-gray-900 text-gray-100">
  <!-- Sidebar -->
  <aside class="w-64 bg-gray-800 border-r border-gray-700">
    <div class="p-6">
      <h1 class="text-2xl font-bold text-white mb-8">RAGEngine</h1>
      
      <nav class="space-y-2">
        <a 
          href="/" 
          class="block px-4 py-2 rounded-lg transition-colors {currentPath === '/' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}"
        >
          🏠 Dashboard
        </a>
        <a 
          href="/search" 
          class="block px-4 py-2 rounded-lg transition-colors {currentPath === '/search' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}"
        >
          🔍 Search
        </a>
        <a 
          href="/documents" 
          class="block px-4 py-2 rounded-lg transition-colors {currentPath === '/documents' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}"
        >
          📄 Documents
        </a>
        <a 
          href="/settings" 
          class="block px-4 py-2 rounded-lg transition-colors {currentPath === '/settings' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}"
        >
          ⚙️ Settings
        </a>
      </nav>
    </div>
    
    <!-- Connection Status -->
    <div class="absolute bottom-4 left-4 right-4">
      <div class="flex items-center space-x-2 text-sm">
        <div class="w-2 h-2 rounded-full {$connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}"></div>
        <span class="text-gray-400">
          {$connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </div>
  </aside>
  
  <!-- Main Content -->
  <main class="flex-1 overflow-auto">
    <slot />
  </main>
</div>
