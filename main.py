# AGENTIC_MIRAI/main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional # Removed Annotated as it wasn't used

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
rag_manager_instance: Optional[RAGManager] = None
try:
    component_factory = ComponentFactory(settings=settings)
    rag_manager_instance = RAGManager(settings=settings, component_factory=component_factory)
except BaseRAGException as e:
    logger.critical(f"Failed to initialize RAGManager: {e}", exc_info=True)
    # rag_manager_instance will remain None, and get_rag_manager will raise 503
except Exception as e: # Catch any other unexpected error during init
    logger.critical(f"Unexpected error during RAGManager initialization: {e}", exc_info=True)


# --- Dependency for RAG Manager ---
def get_rag_manager() -> RAGManager:
    if rag_manager_instance is None:
        logger.error("RAGManager is not available due to initialization failure.")
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
async def http_exception_handler_custom(request, exc: HTTPException): # Renamed to avoid conflict if needed
    # Keep FastAPI's default HTTP exception handling but log it
    logger.warning(f"HTTP Exception from FastAPI: Status {exc.status_code}, Detail: {exc.detail}")
    # Return the original exception's response if needed, or customize
    return JSONResponse(
        status_code=exc.status_code,
        # Pydantic models for error responses could be used here too for consistency
        content={"success": False, "message": "Request error (HTTPException).", "detail": exc.detail}
    )


# --- API Endpoints ---
TEMP_UPLOAD_DIR = "temp_uploads"
# Create directory if it doesn't exist when the module is loaded
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file_body: UploadFileBody = Depends(),
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
        
        # Asynchronously write file to disk to not block event loop if file is large
        # For very large files, consider streaming or background tasks.
        # For typical document sizes, this copyfileobj within an async route
        # will be run in a threadpool by FastAPI.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File '{file.filename}' uploaded to '{temp_file_path}'.")
        logger.info(f"Chunking params: size={file_body.chunk_size}, overlap={file_body.chunk_overlap}")

        # CORRECTED: Call synchronous rag_manager.add_document without await
        nodes_indexed = rag_manager.add_document(
            file_path=temp_file_path,
            doc_metadata={"source_filename": file.filename},
            chunk_size=file_body.chunk_size,
            chunk_overlap=file_body.chunk_overlap
        )
        
        return UploadResponse(
            message="Document processed and indexed successfully.",
            filename=file.filename,
            nodes_indexed=nodes_indexed
        )
    except BaseRAGException as e:
        logger.error(f"Error during document upload processing: {e}", exc_info=True)
        # This will be caught by our rag_exception_handler
        raise e
    except HTTPException: # Re-raise FastAPI's own HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}", exc_info=True)
        # Convert generic exceptions to a standard 500 error response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upload: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.error(f"Error deleting temp file {temp_file_path}: {e}")
        if hasattr(file, 'file') and hasattr(file.file, 'close') and not file.file.closed:
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
        # rag_manager.query() now returns the fully formed QueryResponse object
        # So, just call it and return its result directly.
        query_response_object = rag_manager.query(request) # This is already a QueryResponse
        
        return query_response_object # <<<< CORRECT: Return the object directly
    
    except BaseRAGException as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise e # Re-raise to be caught by your custom handler
    except HTTPException: # Re-raise FastAPI's own HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query: {str(e)}")

# @app.post("/query/", response_model=QueryResponse, tags=["Query"])
# async def query_index(
#     request: QueryRequest,
#     rag_manager: RAGManager = Depends(get_rag_manager)
# ):
#     """
#     Queries the RAG system with the given text and retrieval parameters.
#     `top_k` and `retrieval_strategy` can be specified in the request.
#     """
#     try:
#         # CORRECTED: Call synchronous rag_manager.query without await
#         retrieved_nodes = rag_manager.query(request)
        
#         # Construct the QueryResponse object as expected by response_model
#         return QueryResponse(
#             query_text=request.query_text,
#             retrieved_nodes=retrieved_nodes
#         )
#     except BaseRAGException as e:
#         logger.error(f"Error during query: {e}", exc_info=True)
#         # This will be caught by our rag_exception_handler
#         raise e
#     except HTTPException: # Re-raise FastAPI's own HTTPExceptions
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during query: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query: {str(e)}")


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
        # Assuming rag_manager.clear_index is also synchronous
        rag_manager.clear_index(are_you_sure=True)
        return {"message": f"Index '{settings.INDEX_NAME}' has been cleared and RAG system re-initialized."}
    except BaseRAGException as e:
        logger.error(f"Error clearing index: {e}", exc_info=True)
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error clearing index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while clearing index: {str(e)}")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Agentic Mirai RAG API. See /docs for details."}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Agentic Mirai RAG API on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)