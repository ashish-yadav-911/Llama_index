# AGENTIC_MIRAI/shared/validation/query_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from fastapi import UploadFile, File

class DocumentMetadata(BaseModel):
    source: Optional[str] = None # e.g., filename, URL
    page_number: Optional[int] = None
    extra_info: Optional[Dict[str, Any]] = None

class RetrievedNode(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="The user's query.")
    top_k: Optional[int] = Field(None, gt=0, description="Number of results to retrieve.")
    retrieval_strategy: Optional[Literal["similarity", "mmr"]] = Field(None, description="Retrieval strategy.")
    # Add other configurable parameters like filters, MMR specific params if needed
    # mmr_diversity_bias: Optional[float] = Field(None, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    query_text: str
    retrieved_nodes: List[RetrievedNode]
    synthesized_answer: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    document_id: Optional[str] = None # Or some identifier for the processed document/index
    nodes_indexed: Optional[int] = None

# Not strictly a schema, but useful for FastAPI endpoint
class UploadFileBody:
    def __init__(
        self,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # You could add more metadata here if needed, passed as form data
        # metadata_json: Optional[str] = Form(None) # e.g., "{'category': 'finance'}"
    ):
        self.file = file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.metadata = json.loads(metadata_json) if metadata_json else {}

