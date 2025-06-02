import os
import asyncio
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.multi_collection_knowledge_base_api import MultiCollectionKnowledgeBaseAPI
from src.models.data_models import DocumentCollection
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
app = FastAPI(title="Breath Diagnostics RAGEngine API")

# Configure CORS
origins = [
    "http://localhost:5173",      # SvelteKit dev server
    "http://localhost:4173",      # SvelteKit preview
    "http://localhost:3000",      # Alternative dev port
    "http://frontend:3000",       # Docker container frontend
    "https://your-domain.com",    # Production frontend
]

# Allow Docker network ranges for development
if os.getenv("DOCKER_ENV") == "true":
    origins.extend([
        "http://172.20.0.0/16",   # Docker network range
        "http://ragengine-frontend:3000",  # Container name
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from correct directory (working directory is root)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates from correct directory (working directory is root)
templates = Jinja2Templates(directory="templates")

# Global progress tracking
active_tasks: Dict[str, Dict[str, Any]] = {}
search_tasks: Dict[str, Dict[str, Any]] = {}
search_executor = ThreadPoolExecutor(max_workers=4)
search_events: Dict[str, asyncio.Queue] = {}  # Event queues for SSE updates

# Initialize Multi-Collection RAGEngine API
def get_kb_config():
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "default_collection": os.getenv("DEFAULT_COLLECTION", "current_documents"),
        "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid_hierarchical_semantic"),
        "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
        "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
        "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
        "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
        "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5")),
        "auto_collection_assignment": os.getenv("AUTO_COLLECTION_ASSIGNMENT", "true").lower() == "true"
    }

try:
    kb_api = MultiCollectionKnowledgeBaseAPI(get_kb_config())
    logger.info("Multi-collection RAGEngine API initialized successfully")
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

# Background task cleanup
async def cleanup_old_tasks():
    """Clean up completed tasks after a delay."""
    while True:
        try:
            current_time = datetime.now()
            tasks_to_remove = []
            
            for task_id, task in search_tasks.items():
                # Remove tasks that are completed and older than 5 minutes
                if task.get("status") in ["completed", "error", "cancelled"]:
                    completion_time = task.get("completion_time")
                    if completion_time and (current_time - completion_time).total_seconds() > 300:  # 5 minutes
                        tasks_to_remove.append(task_id)
                
                # Remove very old tasks regardless of status (older than 1 hour)
                start_time = task.get("start_time")
                if start_time and (current_time - start_time).total_seconds() > 3600:  # 1 hour
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                if task_id in search_tasks:
                    logger.info(f"Cleaning up old task: {task_id}")
                    del search_tasks[task_id]
                if task_id in search_events:
                    del search_events[task_id]
            
            await asyncio.sleep(60)  # Run cleanup every minute
        except Exception as e:
            logger.error(f"Error in task cleanup: {e}")
            await asyncio.sleep(60)

# Start background cleanup task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_tasks())

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page with multi-collection support."""
    if not kb_api:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "RAGEngine API not initialized. Check your configuration."
        })
    
    try:
        # Get collection statistics
        collection_stats_result = kb_api.get_collection_statistics()
        collection_stats = collection_stats_result.get("collections", {})
        
        # Get system information
        system_info = {
            "search_config": {
                "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
                "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid_hierarchical_semantic"),
                "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
                "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
                "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5"))
            },
            "collections": {
                "default": os.getenv("DEFAULT_COLLECTION", "current_documents"),
                "available": os.getenv("AVAILABLE_COLLECTIONS", "").split(",") if os.getenv("AVAILABLE_COLLECTIONS") else [],
                "auto_assignment": os.getenv("AUTO_COLLECTION_ASSIGNMENT", "true").lower() == "true"
            }
        }
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "system_info": system_info,
            "collection_stats": collection_stats,
            "api_status": "connected"
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "error": str(e),
            "api_status": "error"
        })

@app.post("/search-with-progress")
async def search_with_progress(
    request: Request, 
    background_tasks: BackgroundTasks,
    query: str = Form(...), 
    mode: str = Form("hybrid"), 
    synthesize: bool = Form(True),
    collections: List[str] = Form(["all"])  # Collection selection
):
    """Initiate search with progress tracking and collection filtering (HTMX version - returns HTML)."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    # Process collection selection
    if "all" in collections or not collections:
        selected_collections = None  # Search all collections
    else:
        selected_collections = []
        for collection_name in collections:
            collection = kb_api.collection_manager.validate_collection(collection_name)
            if collection:
                selected_collections.append(collection_name)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize search tracking
    search_tasks[task_id] = {
        "status": "initializing",
        "progress": 0,
        "phase": "Initializing",
        "message": "Setting up collection search...",
        "query": query,
        "mode": mode,
        "synthesize": synthesize,
        "collections": selected_collections or ["all"],
        "start_time": datetime.now(),
        "estimated_completion": None,
        "current_phase": 1,
        "total_phases": 8 if mode == "hybrid" and synthesize else (6 if synthesize else 4),
        "errors": [],
        "cancelled": False,
        "results": None,
        "results_retrieved": False
    }
    
    # Create event queue for this task
    search_events[task_id] = asyncio.Queue()
    
    logger.info(f"Created collection search task {task_id} for collections: {selected_collections or 'all'}")
    
    # Start background search
    background_tasks.add_task(perform_collection_search_with_progress, task_id, query, mode, synthesize, selected_collections)
    
    # Return progress UI for HTMX
    return templates.TemplateResponse("fragments/search_progress.html", {
        "request": request,
        "task_id": task_id,
        "query": query,
        "mode": mode,
        "synthesize": synthesize
    })

async def perform_collection_search_with_progress(task_id: str, query: str, mode: str, synthesize: bool, collections: Optional[List[str]]):
    """Background task to perform search with detailed progress tracking across collections."""
    logger.info(f"Starting collection search task {task_id} for collections: {collections or 'all'}")
    task = search_tasks[task_id]
    
    try:
        # Phase 1: Initialize (0-10%)
        await update_search_progress(task_id, 1, 5, f"Validating query across collections...")
        await asyncio.sleep(0.3)  # Simulate processing time
        
        if task["cancelled"]:
            return
        
        await update_search_progress(task_id, 1, 10, "Collection search initialized successfully")
        
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
        
        # Phase 3-5: Collection Search (25-70%)
        await update_search_progress(task_id, 3, 30, f"Searching across collections...")
        
        # Perform actual collection search
        start_time = time.time()
        try:
            await update_search_progress(task_id, 4, 50, "Executing multi-collection search...")
            
            results = kb_api.search_collections(
                query=query,
                collections=collections,
                mode=mode,
                synthesize=synthesize
            )
            
            search_time = time.time() - start_time
            await update_search_progress(task_id, 5, 70, f"Multi-collection search completed ({search_time:.1f}s)")
            
        except Exception as e:
            task["errors"].append(f"Collection search failed: {str(e)}")
            await update_search_progress(task_id, 5, 70, "Collection search encountered errors")
            results = {"contexts": [], "total_results": 0, "collections_searched": collections or []}
        
        # Phase 6: Post-processing (70-85%)
        if results.get("contexts") and len(results["contexts"]) > 0:
            await update_search_progress(task_id, 6, 75, "Post-processing results...")
            
            # Simulate post-processing time
            for i in range(3):
                if task["cancelled"]:
                    return
                await asyncio.sleep(0.3)
                progress = 75 + (i + 1) * 3
                await update_search_progress(task_id, 6, progress, f"Analyzing relevance... ({i+1}/3)")
            
            await update_search_progress(task_id, 6, 85, f"Post-processing completed - {len(results['contexts'])} results")
        else:
            await update_search_progress(task_id, 6, 85, "No results found in selected collections")
        
        # Phase 7: Knowledge Synthesis (85-95%) - if requested and results exist
        if synthesize and results.get("synthesized_knowledge"):
            await update_search_progress(task_id, 7, 88, "Performing knowledge synthesis...")
            
            # Synthesis already done in search_collections, just simulate progress
            for i in range(3):
                if task["cancelled"]:
                    return
                await asyncio.sleep(0.5)
                progress = 88 + (i + 1) * 2
                await update_search_progress(task_id, 7, progress, f"Synthesizing knowledge... ({i+1}/3)")
            
            await update_search_progress(task_id, 7, 95, "Knowledge synthesis completed")
        elif synthesize:
            await update_search_progress(task_id, 7, 95, "Synthesis skipped (no results)")
        else:
            await update_search_progress(task_id, 7, 95, "Synthesis not requested")
        
        # Phase 8: Finalize (95-100%)
        await update_search_progress(task_id, 8, 98, "Finalizing results...")
        await asyncio.sleep(0.2)
        
        # Store results
        task["results"] = results
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = f"Search completed! Found {results.get('total_results', 0)} results"
        task["completion_time"] = datetime.now()
        
        # Calculate estimated completion time for future reference
        total_time = (task["completion_time"] - task["start_time"]).total_seconds()
        task["total_time"] = total_time
        
        logger.info(f"Collection search task {task_id} completed successfully in {total_time:.1f}s")
        
        # Clean up after delay
        asyncio.create_task(cleanup_completed_task(task_id, delay=60))
        
    except asyncio.CancelledError:
        task["status"] = "cancelled"
        task["message"] = "Search was cancelled by user"
        task["completion_time"] = datetime.now()
        logger.info(f"Collection search task {task_id} was cancelled")
    except Exception as e:
        logger.error(f"Collection search task {task_id} failed: {e}")
        task["status"] = "error"
        task["message"] = f"Search failed: {str(e)}"
        task["errors"].append(str(e))
        task["completion_time"] = datetime.now()

async def cleanup_completed_task(task_id: str, delay: int = 30):
    """Clean up completed task after a delay."""
    await asyncio.sleep(delay)
    if task_id in search_tasks:
        logger.info(f"Cleaning up completed task {task_id} after {delay}s delay")
        del search_tasks[task_id]
    if task_id in search_events:
        del search_events[task_id]

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
            logger.debug(f"Progress update queued for task {task_id}: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Failed to notify SSE listeners for task {task_id}: {e}")

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
            # Validate task ID
            if not task_id or task_id == "null" or task_id == "undefined":
                logger.warning(f"Invalid task ID received: {task_id}")
                yield f"data: {{\"error\": \"Invalid task ID: {task_id}\"}}\n\n"
                return
            
            # Check if task exists
            if task_id not in search_tasks:
                logger.warning(f"Task {task_id} not found for SSE connection")
                yield f"data: {{\"error\": \"Task not found: {task_id}\"}}\n\n"
                return
            
            logger.info(f"SSE connection established for task {task_id}")
            
            # Send initial connection event
            yield f"data: {{\"status\": \"connected\", \"task_id\": \"{task_id}\"}}\n\n"
            
            # Create event queue if it doesn't exist
            if task_id not in search_events:
                search_events[task_id] = asyncio.Queue()
                logger.info(f"Created event queue for task {task_id}")
            
            event_queue = search_events[task_id]
            
            # Send initial progress if task exists
            if task_id in search_tasks:
                task = search_tasks[task_id]
                data = prepare_progress_data(task)
                yield f"data: {json.dumps(data)}\n\n"
                logger.info(f"Sent initial progress for task {task_id}: {data['progress']}%")
                
                # If task is already completed, send completion event and close SSE
                if task["status"] in ["completed", "error", "cancelled"]:
                    logger.info(f"Task {task_id} already completed, sending completion event")
                    yield f"event: complete\ndata: {json.dumps(data)}\n\n"
                    
                    # Only clean up event queue, keep task for results retrieval
                    if task_id in search_events:
                        del search_events[task_id]
                        logger.info(f"Cleaned up event queue for completed task {task_id}")
                    return
            
            # Better completion handling
            last_progress = -1
            last_message = ""
            max_iterations = 200  # Prevent infinite loops (200 * 0.5s = 100s max)
            iteration = 0
            
            while task_id in search_tasks and iteration < max_iterations:
                try:
                    # Try to get event notification with timeout
                    try:
                        await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        # Continue with polling
                        pass
                    
                    if task_id not in search_tasks:
                        logger.debug(f"Task {task_id} no longer exists, ending SSE stream")
                        break
                        
                    task = search_tasks[task_id]
                    
                    # Send update if progress or message changed
                    if task["progress"] != last_progress or task["message"] != last_message:
                        data = prepare_progress_data(task)
                        yield f"data: {json.dumps(data)}\n\n"
                        logger.debug(f"SSE update sent for task {task_id}: {data['progress']}% - {data['message']}")
                        last_progress = task["progress"]
                        last_message = task["message"]
                    
                    # End stream only when task is truly complete
                    if task["status"] in ["completed", "error", "cancelled"] and task["progress"] >= 100:
                        data = prepare_progress_data(task)
                        yield f"event: complete\ndata: {json.dumps(data)}\n\n"
                        logger.info(f"Task {task_id} completed, sending completion event and closing SSE")
                        
                        # Only clean up event queue, keep task for results retrieval
                        if task_id in search_events:
                            del search_events[task_id]
                            logger.info(f"Cleaned up event queue for task {task_id}")
                        break
                        
                    # Prevent excessive polling
                    await asyncio.sleep(0.5)
                    iteration += 1
                        
                except Exception as inner_e:
                    logger.error(f"Inner SSE loop error for task {task_id}: {inner_e}")
                    break
            
            # Final cleanup check - only event queue
            if iteration >= max_iterations:
                logger.warning(f"Task {task_id} reached max iterations, cleaning up event queue")
                if task_id in search_events:
                    del search_events[task_id]
                    
        except Exception as e:
            logger.error(f"SSE stream error for task {task_id}: {e}")
            yield f"data: {{\"error\": \"Stream error: {str(e)}\"}}\n\n"
        finally:
            # Clean up event queue but keep task for result retrieval
            if task_id in search_events:
                del search_events[task_id]
                logger.debug(f"Cleaned up event queue for task {task_id}")
    
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
        search_tasks[task_id]["completion_time"] = datetime.now()
        return HTMLResponse('<div class="text-yellow-500">Search cancelled</div>')
    return HTMLResponse('<div class="text-red-500">Search not found</div>')

@app.get("/search-results/{task_id}")
async def get_search_results(request: Request, task_id: str):
    """Get the final search results (HTMX version - returns HTML)."""
    logger.info(f"Results requested for task {task_id}")
    
    if task_id not in search_tasks:
        logger.warning(f"Task {task_id} not found when requesting results")
        return HTMLResponse('<div class="text-red-500">Search not found</div>')
    
    task = search_tasks[task_id]
    if task["status"] != "completed" or not task.get("results"):
        logger.warning(f"Results requested for incomplete task {task_id}: status={task['status']}, has_results={task.get('results') is not None}")
        return HTMLResponse('<div class="text-yellow-500">Results not ready</div>')
    
    # Check if results exist
    if not task.get("results"):
        logger.warning(f"Task {task_id} completed but no results found")
        return HTMLResponse('<div class="text-yellow-500">No results available</div>')
    
    # Get results
    results = task["results"]
    query = task["query"]
    mode = task["mode"]
    
    # Mark as retrieved but don't delete immediately
    task["results_retrieved"] = True
    task["results_retrieved_time"] = datetime.now()
    
    logger.info(f"Returning results for task {task_id}: {results.get('total_results', 0)} results")
    
    return templates.TemplateResponse("fragments/search_results.html", {
        "request": request,
        "results": results,
        "query": query,
        "mode": mode
    })

@app.get("/collection-stats")
async def collection_stats(request: Request):
    """Get collection statistics as HTML fragment."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        stats_result = kb_api.get_collection_statistics()
        stats = stats_result.get("collections", {})
        
        return templates.TemplateResponse("collection_stats.html", {
            "request": request,
            "collection_stats": stats
        })
    except Exception as e:
        logger.error(f"Collection stats error: {e}")
        return HTMLResponse(f'<div class="text-red-500">Stats error: {str(e)}</div>')

@app.post("/collection/{collection_name}/clear")
async def clear_collection(request: Request, collection_name: str):
    """Clear a specific collection."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        result = await kb_api.clear_collection(collection_name)
        
        if result.get("status") == "success":
            return HTMLResponse(f'<div class="text-green-600">Collection {collection_name} cleared successfully</div>')
        else:
            return HTMLResponse(f'<div class="text-red-600">Failed to clear collection: {result.get("error", "Unknown error")}</div>')
            
    except Exception as e:
        return HTMLResponse(f'<div class="text-red-500">Clear failed: {str(e)}</div>')

@app.post("/upload-to-collection")
async def upload_to_collection(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    target_collection: str = Form("")
):
    """Upload files with collection assignment."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    task_id = str(uuid.uuid4())
    
    active_tasks[task_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Preparing collection upload...",
        "files": [f.filename for f in files],
        "target_collection": target_collection or "auto-classify",
        "start_time": datetime.now(),
        "errors": []
    }
    
    # Start background processing
    background_tasks.add_task(process_collection_upload, task_id, files, target_collection)
    
    return templates.TemplateResponse("fragments/upload_progress.html", {
        "request": request,
        "task_id": task_id
    })

async def process_collection_upload(task_id: str, files: list[UploadFile], target_collection: str):
    """Process uploads with collection assignment."""
    task = active_tasks[task_id]
    
    try:
        # Save files
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
            
            progress = 10 + (i + 1) / len(files) * 20
            task["progress"] = progress
            task["message"] = f"Saved {file.filename}"
        
        # Process with collection assignment
        task["status"] = "processing"
        task["message"] = "Processing documents with collection assignment..."
        task["progress"] = 30
        
        total_processed = 0
        collection_assignments = {}
        
        for i, file_path in enumerate(saved_files):
            try:
                task["message"] = f"Processing {os.path.basename(file_path)}..."
                
                # Use enhanced ingestion method
                result = await kb_api.ingest_document_to_collection(
                    file_path=file_path,
                    target_collection=target_collection if target_collection else None,
                    auto_classify=not target_collection
                )
                
                if result.get("status") == "success":
                    assigned_collection = result.get("collection", "unknown")
                    collection_assignments[os.path.basename(file_path)] = assigned_collection
                    total_processed += 1
                else:
                    task["errors"].append(f"Failed to process {os.path.basename(file_path)}: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Failed to process {os.path.basename(file_path)}: {str(e)}"
                task["errors"].append(error_msg)
            
            progress = 30 + (i + 1) / len(saved_files) * 60
            task["progress"] = progress
        
        # Complete
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = f"Completed! Processed {total_processed} files"
        task["collection_assignments"] = collection_assignments
        
        if task["errors"]:
            task["message"] += f", {len(task['errors'])} errors"
        
    except Exception as e:
        task["status"] = "error"
        task["message"] = f"Upload failed: {str(e)}"
        task["errors"].append(str(e))
        logger.error(f"Collection upload processing error: {e}")

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
async def system_info(request: Request):
    """Get system information as HTML for HTMX."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        # Get system health status
        health_status = kb_api.vector_db.get_health_status()
        
        system_info = {
            "search_config": {
                "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
                "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid_hierarchical_semantic"),
                "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
                "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
                "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5"))
            },
            "collections": {
                "default": os.getenv("DEFAULT_COLLECTION", "current_documents"),
                "available": os.getenv("AVAILABLE_COLLECTIONS", "").split(",") if os.getenv("AVAILABLE_COLLECTIONS") else [],
                "auto_assignment": os.getenv("AUTO_COLLECTION_ASSIGNMENT", "true").lower() == "true"
            },
            "health": health_status
        }
        
        return templates.TemplateResponse("fragments/system_info.html", {
            "request": request,
            "system_info": system_info
        })
    except Exception as e:
        logger.error(f"System info error: {e}")
        return HTMLResponse(f'<div class="text-red-500">System info error: {str(e)}</div>')

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
            "message": task["message"],
            "start_time": task["start_time"].isoformat(),
            "elapsed_seconds": (datetime.now() - task["start_time"]).total_seconds()
        } for task_id, task in search_tasks.items()},
        "event_queues": list(search_events.keys())
    }

@app.post("/debug/clear-tasks")
async def clear_all_tasks():
    """Debug endpoint to clear all stuck tasks."""
    cleared_tasks = list(search_tasks.keys())
    cleared_events = list(search_events.keys())
    
    search_tasks.clear()
    search_events.clear()
    
    logger.info(f"Cleared {len(cleared_tasks)} tasks and {len(cleared_events)} event queues")
    
    return {
        "message": f"Cleared {len(cleared_tasks)} tasks and {len(cleared_events)} event queues",
        "cleared_tasks": cleared_tasks,
        "cleared_events": cleared_events
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)