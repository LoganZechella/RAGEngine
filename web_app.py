import os
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

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
app = FastAPI(title="RAGEngine Web Interface")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global progress tracking
active_tasks: Dict[str, Dict[str, Any]] = {}

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

@app.post("/search")
async def search(request: Request, query: str = Form(...), mode: str = Form("hybrid"), synthesize: bool = Form(True)):
    """Handle search requests and return results fragment."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        # Determine search function based on mode
        if mode == "dense":
            results = kb_api.dense_search(query, top_k=10)
            results["synthesis"] = None
        elif mode == "sparse":
            results = kb_api.sparse_search(query, top_k=10)
            results["synthesis"] = None
        else:  # hybrid
            results = kb_api.search(query, synthesize=synthesize)
        
        return templates.TemplateResponse("fragments/search_results.html", {
            "request": request,
            "results": results,
            "query": query,
            "mode": mode
        })
    except Exception as e:
        logger.error(f"Search error: {e}")
        return HTMLResponse(f'<div class="text-red-500">Search failed: {str(e)}</div>')

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
async def system_info(request: Request):
    """Get system information fragment."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        info = kb_api.get_system_info()
        return templates.TemplateResponse("fragments/system_info.html", {
            "request": request,
            "system_info": info
        })
    except Exception as e:
        return HTMLResponse(f'<div class="text-red-500">System info error: {str(e)}</div>')

@app.get("/documents")
async def document_list(request: Request):
    """Get processed documents list."""
    if not kb_api:
        return HTMLResponse('<div class="text-red-500">API not available</div>')
    
    try:
        docs = kb_api.get_processed_documents()
        return templates.TemplateResponse("fragments/document_list.html", {
            "request": request,
            "documents": docs
        })
    except Exception as e:
        return HTMLResponse(f'<div class="text-red-500">Document list error: {str(e)}</div>')

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 