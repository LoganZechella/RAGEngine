import os
import asyncio
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.knowledge_base_api import KnowledgeBaseAPI
from dotenv import load_dotenv
from loguru import logger

# Load environment
load_dotenv()

# Custom EventSourceResponse implementation
class EventSourceResponse(StreamingResponse):
    def __init__(self, generator, **kwargs):
        super().__init__(
            content=generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
            **kwargs
        )

# Initialize FastAPI app
app = FastAPI(title="RAGEngine API")

# Configure CORS for SvelteKit frontend
origins = [
    "http://localhost:5173",      # SvelteKit dev server
    "http://localhost:4173",      # SvelteKit preview
    "http://localhost:3000",      # Alternative dev port
    "https://your-domain.com",    # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global progress tracking
active_tasks: Dict[str, Dict[str, Any]] = {}

# Add after existing global variables
search_tasks: Dict[str, Dict[str, Any]] = {}
search_executor = ThreadPoolExecutor(max_workers=4)
search_events: Dict[str, asyncio.Queue] = {}  # Event queues for SSE updates

# Initialize RAGEngine API
def get_kb_config():
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),
        "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid_hierarchical_semantic"),
        "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
        "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
        "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
        "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
        "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5"))
    }

try:
    kb_api = KnowledgeBaseAPI(get_kb_config())
    logger.info("RAGEngine API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAGEngine API: {e}")
    kb_api = None

# Jinja2 filters for data formatting
def format_confidence(value):
    if value is None:
        return "Unknown"
    elif value >= 0.8:
        return "High"
    elif value >= 0.6:
        return "Moderate"
    elif value >= 0.4:
        return "Low"
    else:
        return "Very Low"

def format_score_bar(score, max_width=10):
    if score is None:
        return "░" * max_width
    filled = int(score * max_width)
    return "█" * filled + "░" * (max_width - filled)

def format_timestamp(timestamp):
    if isinstance(timestamp, str):
        return timestamp[:19]  # Truncate to YYYY-MM-DD HH:MM:SS
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"

# Register filters
templates.env.filters["format_confidence"] = format_confidence
templates.env.filters["format_score_bar"] = format_score_bar
templates.env.filters["format_timestamp"] = format_timestamp

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    if not kb_api:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "RAGEngine API not initialized. Check your configuration."
        })
    
    try:
        system_info = kb_api.get_system_info()
        processed_docs = kb_api.get_processed_documents()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "system_info": system_info,
            "doc_count": len(processed_docs),
            "api_status": "connected"
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "error": str(e),
            "api_status": "error"
        })

# Add new search endpoints (replace existing @app.post("/search"))
@app.post("/search-with-progress")
async def search_with_progress(
    request: Request, 
    background_tasks: BackgroundTasks,
    query: str = Form(...), 
    mode: str = Form("hybrid"), 
    synthesize: bool = Form(True)
):
    """Initiate search with progress tracking."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize search tracking
    search_tasks[task_id] = {
        "status": "initializing",
        "progress": 0,
        "phase": "Initializing",
        "message": "Setting up search parameters...",
        "query": query,
        "mode": mode,
        "synthesize": synthesize,
        "start_time": datetime.now(),
        "estimated_completion": None,
        "current_phase": 1,
        "total_phases": 8 if mode == "hybrid" and synthesize else (6 if synthesize else 4),
        "errors": [],
        "cancelled": False,
        "results": None
    }
    
    # Create event queue for this task
    search_events[task_id] = asyncio.Queue()
    
    logger.info(f"Created search task {task_id} for query: '{query}' mode: {mode}")
    
    # Start background search
    background_tasks.add_task(perform_search_with_progress, task_id, query, mode, synthesize)
    
    # Return progress UI
    return templates.TemplateResponse("fragments/search_progress.html", {
        "request": request,
        "task_id": task_id,
        "query": query,
        "mode": mode,
        "synthesize": synthesize
    })

async def perform_search_with_progress(task_id: str, query: str, mode: str, synthesize: bool):
    """Background task to perform search with detailed progress tracking."""
    logger.info(f"Starting search task {task_id}")
    task = search_tasks[task_id]
    
    try:
        # Phase 1: Initialize (0-10%)
        await update_search_progress(task_id, 1, 5, "Validating query parameters...")
        await asyncio.sleep(0.3)  # Simulate processing time
        
        if task["cancelled"]:
            return
        
        await update_search_progress(task_id, 1, 10, "Search initialized successfully")
        
        # Phase 2: Embedding Generation (10-25%) - for dense/hybrid only
        if mode in ["dense", "hybrid"]:
            await update_search_progress(task_id, 2, 15, "Generating query embeddings...")
            
            # Simulate embedding generation time
            for i in range(3):
                if task["cancelled"]:
                    return
                await asyncio.sleep(0.4)
                progress = 15 + (i + 1) * 3
                await update_search_progress(task_id, 2, progress, f"Processing embedding vectors... ({i+1}/3)")
            
            await update_search_progress(task_id, 2, 25, "Query embeddings generated")
        
        # Phase 3: Vector Search (25-45%)
        if mode in ["dense", "hybrid"]:
            await update_search_progress(task_id, 3, 30, "Searching vector database...")
            
            # Perform actual dense search
            start_time = time.time()
            try:
                if mode == "dense":
                    results = kb_api.dense_search(query, top_k=10)
                    results["synthesis"] = None
                else:
                    # For hybrid, we'll do the full search later
                    pass
                
                search_time = time.time() - start_time
                await update_search_progress(task_id, 3, 45, f"Vector search completed ({search_time:.1f}s)")
                
            except Exception as e:
                task["errors"].append(f"Vector search failed: {str(e)}")
                await update_search_progress(task_id, 3, 45, "Vector search encountered errors")
        
        # Phase 4: Keyword Search (45-60%) - hybrid only
        if mode == "hybrid":
            await update_search_progress(task_id, 4, 50, "Performing keyword search...")
            await asyncio.sleep(0.6)  # Simulate keyword search
            await update_search_progress(task_id, 4, 60, "Keyword search completed")
        
        # Phase 5: Result Fusion (60-70%) - hybrid only
        if mode == "hybrid":
            await update_search_progress(task_id, 5, 65, "Combining search results...")
            await asyncio.sleep(0.3)
            await update_search_progress(task_id, 5, 70, "Results combined using RRF")
        
        # For hybrid mode, perform the actual search now
        if mode == "hybrid":
            await update_search_progress(task_id, 5, 70, "Executing hybrid search...")
            try:
                results = kb_api.search(query, synthesize=False)  # Without synthesis for now
            except Exception as e:
                task["errors"].append(f"Hybrid search failed: {str(e)}")
                results = {"contexts": [], "num_results": 0}
        elif mode == "sparse":
            await update_search_progress(task_id, 3, 45, "Performing sparse search...")
            try:
                results = kb_api.sparse_search(query, top_k=10)
                results["synthesis"] = None
            except Exception as e:
                task["errors"].append(f"Sparse search failed: {str(e)}")
                results = {"contexts": [], "num_results": 0}
        
        # Phase 6: Re-ranking (70-85%) - if results exist
        if results.get("contexts") and len(results["contexts"]) > 0:
            await update_search_progress(task_id, 6, 75, "Re-ranking results for relevance...")
            
            # Simulate re-ranking time (actual re-ranking already happened in kb_api.search)
            for i in range(4):
                if task["cancelled"]:
                    return
                await asyncio.sleep(0.5)
                progress = 75 + (i + 1) * 2
                await update_search_progress(task_id, 6, progress, f"Analyzing relevance... ({i+1}/4)")
            
            await update_search_progress(task_id, 6, 85, f"Re-ranking completed ({len(results['contexts'])} results)")
        else:
            await update_search_progress(task_id, 6, 85, "No results to re-rank")
        
        # Phase 7: Knowledge Synthesis (85-95%) - if enabled
        if synthesize and results.get("contexts"):
            await update_search_progress(task_id, 7, 87, "Initializing AI analysis...")
            
            try:
                # Perform synthesis
                await update_search_progress(task_id, 7, 90, "Generating knowledge synthesis...")
                synthesis_results = kb_api.search(query, synthesize=True)
                results["synthesis"] = synthesis_results.get("synthesis")
                
                await update_search_progress(task_id, 7, 95, "Knowledge synthesis completed")
                
            except Exception as e:
                task["errors"].append(f"Knowledge synthesis failed: {str(e)}")
                await update_search_progress(task_id, 7, 95, "Synthesis failed, continuing...")
        elif synthesize:
            await update_search_progress(task_id, 7, 95, "No content available for synthesis")
        
        # Phase 8: Finalize (95-100%)
        await update_search_progress(task_id, 8, 97, "Finalizing results...")
        await asyncio.sleep(0.2)
        
        # Store results
        task["results"] = results
        task["status"] = "completed"
        task["progress"] = 100
        task["phase"] = "Completed"
        task["message"] = f"Search completed! Found {results.get('num_results', 0)} results"
        task["estimated_completion"] = datetime.now()
        
        # Notify completion
        if task_id in search_events:
            try:
                await search_events[task_id].put("complete")
            except Exception as e:
                logger.error(f"Failed to notify completion for task {task_id}: {e}")
        
    except Exception as e:
        logger.error(f"Search task {task_id} failed: {e}")
        task["status"] = "error"
        task["message"] = f"Search failed: {str(e)}"
        task["errors"].append(str(e))
        
        # Notify error
        if task_id in search_events:
            try:
                await search_events[task_id].put("error")
            except Exception as e:
                logger.error(f"Failed to notify error for task {task_id}: {e}")

async def update_search_progress(task_id: str, phase: int, progress: int, message: str):
    """Update search progress for a task."""
    if task_id not in search_tasks:
        return
    
    task = search_tasks[task_id]
    if task["cancelled"]:
        return
    
    task["current_phase"] = phase
    task["progress"] = progress
    task["message"] = message
    task["phase"] = get_phase_name(phase)
    
    # Estimate completion time
    elapsed = (datetime.now() - task["start_time"]).total_seconds()
    if progress > 10:
        estimated_total = elapsed * (100 / progress)
        estimated_remaining = estimated_total - elapsed
        task["estimated_completion"] = datetime.now().timestamp() + estimated_remaining
    
    # Notify SSE listeners
    if task_id in search_events:
        try:
            await search_events[task_id].put("update")
            logger.debug(f"Progress update sent for task {task_id}: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Failed to notify SSE listeners for task {task_id}: {e}")
    else:
        logger.warning(f"No event queue found for task {task_id}")

def get_phase_name(phase: int) -> str:
    """Get human-readable phase name."""
    phase_names = {
        1: "Initializing",
        2: "Embedding Generation", 
        3: "Vector Search",
        4: "Keyword Search",
        5: "Result Fusion",
        6: "Re-ranking",
        7: "Knowledge Synthesis",
        8: "Finalizing"
    }
    return phase_names.get(phase, "Processing")

@app.get("/search-progress/{task_id}")
async def search_progress_stream(task_id: str):
    """Server-Sent Events endpoint for search progress."""
    async def event_generator():
        try:
            # Send initial connection event
            yield "event: connected\ndata: {\"status\": \"connected\"}\n\n"
            
            # Create event queue if it doesn't exist
            if task_id not in search_events:
                search_events[task_id] = asyncio.Queue()
            
            event_queue = search_events[task_id]
            
            # Send initial progress if task exists
            if task_id in search_tasks:
                task = search_tasks[task_id]
                data = prepare_progress_data(task)
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            
            # Listen for updates with polling fallback
            last_progress = -1
            while task_id in search_tasks:
                try:
                    # Try to get event notification with short timeout
                    try:
                        await asyncio.wait_for(event_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass  # Continue with polling
                    
                    if task_id not in search_tasks:
                        break
                        
                    task = search_tasks[task_id]
                    
                    # Only send update if progress changed
                    if task["progress"] != last_progress:
                        data = prepare_progress_data(task)
                        yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                        last_progress = task["progress"]
                    
                    # Check if task is complete
                    if task["status"] in ["completed", "error", "cancelled"]:
                        data = prepare_progress_data(task)
                        yield f"event: complete\ndata: {json.dumps(data)}\n\n"
                        break
                        
                    # Small delay to prevent excessive polling
                    await asyncio.sleep(0.5)
                        
                except Exception as inner_e:
                    logger.error(f"Inner SSE loop error for task {task_id}: {inner_e}")
                    break
                    
        except Exception as e:
            logger.error(f"SSE stream error for task {task_id}: {e}")
            yield f"event: error\ndata: {{\"error\": \"Stream error: {str(e)}\"}}\n\n"
        finally:
            # Clean up event queue
            if task_id in search_events:
                del search_events[task_id]
    
    return EventSourceResponse(event_generator())

def prepare_progress_data(task: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare progress data for SSE transmission."""
    elapsed = (datetime.now() - task["start_time"]).total_seconds()
    estimated_remaining = None
    if task.get("estimated_completion") and task["progress"] < 100:
        estimated_remaining = max(0, task["estimated_completion"] - datetime.now().timestamp())
    
    return {
        "progress": task["progress"],
        "status": task["status"],
        "phase": task.get("phase", "Processing"),
        "message": task["message"],
        "current_phase": task["current_phase"],
        "total_phases": task["total_phases"],
        "elapsed_time": round(elapsed, 1),
        "estimated_remaining": round(estimated_remaining, 1) if estimated_remaining else None,
        "errors": task["errors"],
        "results_available": task.get("results") is not None,
        "query": task.get("query", "")
    }

@app.post("/cancel-search/{task_id}")
async def cancel_search(task_id: str):
    """Cancel a running search task."""
    if task_id in search_tasks:
        search_tasks[task_id]["cancelled"] = True
        search_tasks[task_id]["status"] = "cancelled"
        search_tasks[task_id]["message"] = "Search cancelled by user"
        return HTMLResponse('<div class="text-yellow-500">Search cancelled</div>')
    return HTMLResponse('<div class="text-red-500">Search not found</div>')

@app.get("/search-results/{task_id}")
async def get_search_results(request: Request, task_id: str):
    """Get the final search results."""
    if task_id not in search_tasks:
        return HTMLResponse('<div class="text-red-500">Search not found</div>')
    
    task = search_tasks[task_id]
    if task["status"] != "completed" or not task.get("results"):
        return HTMLResponse('<div class="text-yellow-500">Results not ready</div>')
    
    # Clean up task after retrieving results
    results = task["results"]
    query = task["query"]
    mode = task["mode"]
    
    # Remove task from memory
    del search_tasks[task_id]
    
    return templates.TemplateResponse("fragments/search_results.html", {
        "request": request,
        "results": results,
        "query": query,
        "mode": mode
    })

# JSON API Endpoints for SvelteKit Frontend

@app.post("/api/search")
async def api_search(
    query: str = Form(...),
    mode: str = Form("hybrid"),
    synthesize: bool = Form(True)
):
    """JSON API endpoint for search functionality."""
    if not kb_api:
        raise HTTPException(status_code=503, detail="API not available")
    
    try:
        logger.info(f"API search request - Query: '{query}', Mode: {mode}, Synthesize: {synthesize}")
        
        if mode == "dense":
            results = kb_api.dense_search(query, top_k=10)
            if synthesize and results.get("contexts"):
                # Add synthesis for dense search
                full_results = kb_api.search(query, synthesize=True)
                results["synthesis"] = full_results.get("synthesis")
        elif mode == "sparse":
            results = kb_api.sparse_search(query, top_k=10)
            if synthesize and results.get("contexts"):
                # Add synthesis for sparse search
                full_results = kb_api.search(query, synthesize=True)
                results["synthesis"] = full_results.get("synthesis")
        else:  # hybrid
            results = kb_api.search(query, synthesize=synthesize)
        
        # Ensure consistent response format
        if "contexts" not in results:
            results["contexts"] = []
        if "num_results" not in results:
            results["num_results"] = len(results.get("contexts", []))
        
        # Transform contexts to match frontend expectations
        transformed_contexts = []
        for ctx in results.get("contexts", []):
            transformed_ctx = {
                "content": ctx.get("text", ""),
                "score": ctx.get("score", ctx.get("rerank_score", ctx.get("initial_score", 0))),
                "metadata": {
                    "filename": ctx.get("metadata", {}).get("filename", "Unknown"),
                    "page": ctx.get("metadata", {}).get("page"),
                    "chunk_id": ctx.get("chunk_id"),
                    **ctx.get("metadata", {})
                }
            }
            transformed_contexts.append(transformed_ctx)
        
        results["contexts"] = transformed_contexts
        
        logger.info(f"Search completed - Found {results['num_results']} results")
        return results
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
):
    """Handle file uploads with background processing."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    active_tasks[task_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Preparing upload...",
        "files": [f.filename for f in files],
        "start_time": datetime.now(),
        "errors": []
    }
    
    # Start background processing
    background_tasks.add_task(process_uploaded_files, task_id, files)
    
    return templates.TemplateResponse("fragments/upload_progress.html", {
        "request": request,
        "task_id": task_id
    })

async def process_uploaded_files(task_id: str, files: list[UploadFile]):
    """Background task to process uploaded files."""
    task = active_tasks[task_id]
    
    try:
        # Save uploaded files
        saved_files = []
        upload_dir = "./uploaded_documents"
        os.makedirs(upload_dir, exist_ok=True)
        
        task["status"] = "saving"
        task["message"] = "Saving uploaded files..."
        task["progress"] = 10
        
        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file_path)
            
            progress = 10 + (i + 1) / len(files) * 20  # 10-30% for saving
            task["progress"] = progress
            task["message"] = f"Saved {file.filename}"
        
        # Process documents
        task["status"] = "processing"
        task["message"] = "Processing documents..."
        task["progress"] = 30
        
        total_processed = 0
        total_errors = 0
        
        for i, file_path in enumerate(saved_files):
            try:
                task["message"] = f"Processing {os.path.basename(file_path)}..."
                stats = kb_api.ingest_single_document(file_path)
                
                if stats["errors"]:
                    task["errors"].extend(stats["errors"])
                    total_errors += 1
                else:
                    total_processed += 1
                
            except Exception as e:
                error_msg = f"Failed to process {os.path.basename(file_path)}: {str(e)}"
                task["errors"].append(error_msg)
                total_errors += 1
            
            # Update progress (30-90% for processing)
            progress = 30 + (i + 1) / len(saved_files) * 60
            task["progress"] = progress
        
        # Complete
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = f"Completed! Processed {total_processed} files successfully"
        if total_errors > 0:
            task["message"] += f", {total_errors} errors"
        
    except Exception as e:
        task["status"] = "error"
        task["message"] = f"Upload failed: {str(e)}"
        task["errors"].append(str(e))
        logger.error(f"Upload processing error: {e}")

@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    """Server-Sent Events endpoint for progress updates."""
    async def event_generator():
        while task_id in active_tasks:
            task = active_tasks[task_id]
            
            # Send progress update
            data = {
                "progress": task["progress"],
                "status": task["status"],
                "message": task["message"],
                "errors": task["errors"]
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # Clean up completed tasks after sending final update
            if task["status"] in ["completed", "error"]:
                await asyncio.sleep(1)  # Give client time to receive final update
                if task_id in active_tasks:
                    del active_tasks[task_id]
                break
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    return EventSourceResponse(event_generator())

@app.get("/system-info")
async def system_info():
    """Get system information as JSON."""
    if not kb_api:
        raise HTTPException(status_code=503, detail="API not available")
    
    try:
        info = kb_api.get_system_info()
        return info
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")

@app.get("/documents")
async def document_list():
    """Get processed documents list as JSON."""
    if not kb_api:
        raise HTTPException(status_code=503, detail="API not available")
    
    try:
        docs = kb_api.get_processed_documents()
        return docs
    except Exception as e:
        logger.error(f"Document list error: {e}")
        raise HTTPException(status_code=500, detail=f"Document list error: {str(e)}")

@app.post("/collection/{action}")
async def collection_action(request: Request, action: str):
    """Handle collection management actions."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        if action == "clear":
            success = kb_api.clear_collection()
            message = "Collection cleared successfully" if success else "Failed to clear collection"
        elif action == "delete":
            success = kb_api.delete_collection()
            message = "Collection deleted successfully" if success else "Failed to delete collection"
        elif action == "recreate":
            success = kb_api.recreate_collection()
            message = "Collection recreated successfully" if success else "Failed to recreate collection"
        else:
            return HTMLResponse('<div class="text-red-500">Invalid action</div>')
        
        status_class = "text-green-600" if success else "text-red-600"
        return HTMLResponse(f'<div class="{status_class}">{message}</div>')
        
    except Exception as e:
        return HTMLResponse(f'<div class="text-red-500">Action failed: {str(e)}</div>')

@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors."""
    return Response(content="", media_type="image/x-icon")

@app.get("/debug/search-tasks")
async def debug_search_tasks():
    """Debug endpoint to check active search tasks."""
    return {
        "active_tasks": list(search_tasks.keys()),
        "task_details": {task_id: {
            "status": task["status"],
            "progress": task["progress"],
            "phase": task.get("phase", "Unknown"),
            "message": task["message"]
        } for task_id, task in search_tasks.items()},
        "event_queues": list(search_events.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 