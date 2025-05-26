<script lang="ts">
  import { onMount } from 'svelte';
  import { documents, connectionStatus } from '$lib/stores/system';
  import { api } from '$lib/api';

  let loading = true;
  let error: string | null = null;
  let uploading = false;
  let uploadError: string | null = null;
  let uploadSuccess = false;

  onMount(async () => {
    await loadDocuments();
  });

  async function loadDocuments() {
    loading = true;
    error = null;
    
    try {
      const docs = await api.getDocuments();
      documents.set(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
      error = err instanceof Error ? err.message : 'Failed to load documents';
    } finally {
      loading = false;
    }
  }

  async function handleFileUpload(event: Event) {
    const target = event.target as HTMLInputElement;
    const files = target.files;
    
    if (!files || files.length === 0) return;
    
    uploading = true;
    uploadError = null;
    uploadSuccess = false;
    
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      
      await api.uploadFiles(formData);
      uploadSuccess = true;
      
      // Reload documents after successful upload
      setTimeout(() => {
        loadDocuments();
        uploadSuccess = false;
      }, 2000);
      
    } catch (err) {
      console.error('Upload failed:', err);
      uploadError = err instanceof Error ? err.message : 'Upload failed';
    } finally {
      uploading = false;
      // Reset file input
      target.value = '';
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatDate(dateString: string): string {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  }
</script>

<div class="p-6">
  <div class="mb-8">
    <h2 class="text-3xl font-bold text-white mb-2">Documents</h2>
    <p class="text-gray-400">Manage your knowledge base documents</p>
  </div>

  <!-- File Upload Section -->
  <div class="card mb-8">
    <h3 class="text-xl font-semibold text-white mb-4">📁 Upload Documents</h3>
    
    <div class="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
      <input 
        type="file" 
        multiple 
        accept=".pdf,.txt,.html,.md"
        class="hidden" 
        id="file-input"
        on:change={handleFileUpload}
        disabled={uploading}
      >
      <label for="file-input" class="cursor-pointer">
        <div class="text-6xl mb-4">📎</div>
        <p class="text-gray-300 mb-2">
          {uploading ? 'Uploading...' : 'Click to select files or drag and drop'}
        </p>
        <p class="text-gray-500 text-sm">Supports PDF, TXT, HTML, MD files</p>
      </label>
    </div>
    
    {#if uploadError}
      <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mt-4">
        <strong>Upload Error:</strong> {uploadError}
      </div>
    {/if}
    
    {#if uploadSuccess}
      <div class="bg-green-900 border border-green-700 text-green-200 px-4 py-3 rounded mt-4">
        <strong>Success:</strong> Files uploaded successfully!
      </div>
    {/if}
  </div>

  <!-- Documents List -->
  <div class="card">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-xl font-semibold text-white">📚 Processed Documents</h3>
      <button 
        on:click={loadDocuments}
        class="btn-secondary"
        disabled={loading}
      >
        {loading ? '⏳' : '🔄'} Refresh
      </button>
    </div>
    
    {#if error}
      <div class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded mb-4">
        <strong>Error:</strong> {error}
      </div>
    {/if}

    {#if loading}
      <div class="animate-pulse space-y-3">
        <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
        <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
        <div class="bg-gray-700 rounded-lg p-4 h-20"></div>
      </div>
    {:else if $documents && $documents.length > 0}
      <div class="space-y-3">
        {#each $documents as doc}
          <div class="bg-gray-700 rounded-lg p-4">
            <div class="flex items-center justify-between">
              <div class="flex-1">
                <div class="flex items-center space-x-3">
                  <span class="text-2xl">📄</span>
                  <div>
                    <h4 class="font-semibold text-white">{doc.filename || doc.title || 'Untitled Document'}</h4>
                    <div class="text-sm text-gray-400 space-x-4">
                      <span>{doc.chunks || 0} chunks</span>
                      {#if doc.file_size}
                        <span>{formatFileSize(doc.file_size)}</span>
                      {/if}
                      {#if doc.upload_date}
                        <span>Uploaded: {formatDate(doc.upload_date)}</span>
                      {/if}
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="flex items-center space-x-2">
                {#if doc.status === 'processed'}
                  <span class="bg-green-600 text-green-100 px-2 py-1 rounded text-xs">✓ Processed</span>
                {:else if doc.status === 'processing'}
                  <span class="bg-yellow-600 text-yellow-100 px-2 py-1 rounded text-xs">⏳ Processing</span>
                {:else if doc.status === 'error'}
                  <span class="bg-red-600 text-red-100 px-2 py-1 rounded text-xs">❌ Error</span>
                {:else}
                  <span class="bg-gray-600 text-gray-100 px-2 py-1 rounded text-xs">❓ Unknown</span>
                {/if}
              </div>
            </div>
            
            {#if doc.error_message}
              <div class="mt-2 text-sm text-red-400">
                Error: {doc.error_message}
              </div>
            {/if}
          </div>
        {/each}
      </div>
      
      <div class="mt-6 text-center text-gray-400">
        Total: {$documents.length} documents
      </div>
    {:else}
      <div class="text-center text-gray-400 py-8">
        <div class="text-6xl mb-4">📚</div>
        <p class="text-lg mb-2">No documents found</p>
        <p class="text-sm">Upload some documents to get started with your knowledge base.</p>
      </div>
    {/if}
  </div>
</div> 