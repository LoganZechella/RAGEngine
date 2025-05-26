import { onDestroy } from 'svelte';

interface SSEOptions {
  onOpen?: () => void;
  onMessage?: (data: any) => void;
  onError?: (error: Event | Error) => void;
}

interface SSEConnection {
  connect: () => void;
  disconnect: () => void;
  cleanup: () => void;
}

export function createSSEConnection(url: string, options: SSEOptions = {}): SSEConnection {
  let eventSource: EventSource | null = null;
  let mounted = true;

  const connect = () => {
    if (!mounted) return;
    
    eventSource = new EventSource(url);
    
    eventSource.onopen = () => {
      console.log('SSE connection opened:', url);
      options.onOpen?.();
    };
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        options.onMessage?.(data);
      } catch (error) {
        console.error('Failed to parse SSE data:', error);
        options.onError?.(error as Error);
      }
    };
    
    eventSource.onerror = (event) => {
      console.error('SSE connection error:', event);
      options.onError?.(event);
    };
  };

  const disconnect = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  };

  const cleanup = () => {
    mounted = false;
    disconnect();
  };

  // Auto-cleanup on component destroy
  onDestroy(cleanup);

  return { connect, disconnect, cleanup };
} 