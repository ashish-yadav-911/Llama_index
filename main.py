# AGENTIC_MIRAI/main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional, Annotated

from shared.config import settings
from shared.log import get_logger
from shared.exceptions import BaseRAGException
from shared.validation.query_schema import QueryRequest, QueryResponse, UploadResponse, UploadFileBody
from app.factory.component_factory import ComponentFactory
from app.modules.rag_manager import RAGManager

logger = get_logger(__name__)

app = FastAPI(
    title="Agentic Mirai RAG API",
    description="A modular RAG system API",
    version="0.1.0"
)

# --- Global RAG Manager ---
# This creates a single RAGManager instance when the app starts.
# Be mindful of state if you have multiple workers in production (Gunicorn, Uvicorn workers)
# For SimpleVectorStore (in-memory), each worker would have its own copy unless persisted and shared.
# For external vector stores (Pinecone, PGVector), this is generally fine.
try:
    component_factory = ComponentFactory(settings=settings)
    rag_manager_instance = RAGManager(settings=settings, component_factory=component_factory)
except BaseRAGException as e:
    logger.critical(f"Failed to initialize RAGManager: {e}", exc_info=True)
    # Exit if critical components fail to load, or handle gracefully
    # For now, we'll let it run and endpoints will fail if rag_manager_instance is None
    # A better approach might be to prevent FastAPI from starting.
    rag_manager_instance = None # type: ignore 
    # raise RuntimeError(f"Critical RAGManager initialization failure: {e}") from e


# --- Dependency for RAG Manager ---
def get_rag_manager() -> RAGManager:
    if rag_manager_instance is None:
        logger.error("RAGManager is not available.")
        raise HTTPException(status_code=503, detail="RAG system is not initialized or currently unavailable.")
    return rag_manager_instance

# --- Exception Handlers ---
@app.exception_handler(BaseRAGException)
async def rag_exception_handler(request, exc: BaseRAGException):
    logger.error(f"RAG Exception: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500, # Or map specific exceptions to different codes
        content={"success": False, "message": "An internal RAG error occurred.", "detail": exc.message},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # Keep FastAPI's default HTTP exception handling but log it
    logger.warning(f"HTTP Exception: Status {exc.status_code}, Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": "Request error.", "detail": exc.detail} # Standardize error response
    )


# --- API Endpoints ---
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file_body: UploadFileBody = Depends(), # Use Depends for form data + file
    rag_manager: RAGManager = Depends(get_rag_manager)
):
    """
    Uploads a document for processing and indexing.
    You can specify `chunk_size` and `chunk_overlap` as form fields.
    """
    file = file_body.file
    temp_file_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File '{file.filename}' uploaded to '{temp_file_path}'.")
        logger.info(f"Chunking params: size={file_body.chunk_size}, overlap={file_body.chunk_overlap}")

        # Here you could extract more metadata if passed in UploadFileBody
        # doc_metadata = file_body.metadata 

        nodes_indexed = rag_manager.add_document(
            file_path=temp_file_path,
            doc_metadata={"source_filename": file.filename}, # Basic metadata
            chunk_size=file_body.chunk_size, # Pass frontend configured params
            chunk_overlap=file_body.chunk_overlap
        )
        
        return UploadResponse(
            message="Document processed and indexed successfully.",
            filename=file.filename,
            nodes_indexed=nodes_indexed
        )
    except BaseRAGException as e: # Catch our custom exceptions
        logger.error(f"Error during document upload processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if hasattr(file, 'file') and hasattr(file.file, 'close'): # Ensure file is closed
             file.file.close()


@app.post("/query/", response_model=QueryResponse, tags=["Query"])
async def query_index(
    request: QueryRequest,
    rag_manager: RAGManager = Depends(get_rag_manager)
):
    """
    Queries the RAG system with the given text and retrieval parameters.
    `top_k` and `retrieval_strategy` can be specified in the request.
    """
    try:
        retrieved_nodes = rag_manager.query(request)
        return QueryResponse(
            query_text=request.query_text,
            retrieved_nodes=retrieved_nodes
        )
    except BaseRAGException as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query: {str(e)}")

@app.post("/admin/clear_index/", status_code=200, tags=["Admin"])
async def clear_all_data(
    confirmation: bool = Form(..., description="Must be true to confirm deletion."),
    rag_manager: RAGManager = Depends(get_rag_manager)
):
    """
    Clears all data from the index. This is a destructive operation.
    Requires `confirmation=true` in the form data.
    """
    if not confirmation:
        raise HTTPException(status_code=400, detail="Deletion not confirmed. Send 'confirmation=true'.")
    try:
        rag_manager.clear_index(are_you_sure=True)
        return {"message": f"Index '{settings.INDEX_NAME}' has been cleared and RAG system re-initialized."}
    except BaseRAGException as e:
        logger.error(f"Error clearing index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error clearing index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Agentic Mirai RAG API. See /docs for details."}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Agentic Mirai RAG API on {settings.API_HOST}:{settings.API_PORT}")
    # OLD WAY: uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
    # NEW WAY FOR RELOAD:
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)