# AGENTIC_MIRAI/app/frameworks/llama_index/loaders.py
from typing import List, Any, Optional, Dict
from llama_index.core.schema import Document # type: ignore
from llama_index.core import SimpleDirectoryReader # type: ignore
#from llama_index.core import  # type: ignore
#from llama_index.readers.file import SimpleDirectoryReader # Example readers
# from llama_index.readers.web import SimpleWebPageReader # Example for web
from app.abstract.loader import BaseDocumentLoader
from shared.exceptions import DocumentProcessingError
from shared.log import get_logger
import os

logger = get_logger(__name__)

class LlamaIndexDocumentLoader(BaseDocumentLoader):
    def load(self, source: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info(f"Loading document from source: {source} with LlamaIndex")
        try:
            # Basic type sniffing based on source
            if isinstance(source, str):
                if source.startswith("http://") or source.startswith("https://"):
                    # return SimpleWebPageReader(html_to_text=True).load_data([source])
                    raise NotImplementedError("Web reader not fully implemented yet for this generic loader.")
                elif os.path.isfile(source):
                    if source.lower().endswith(".pdf"):
                        return SimpleDirectoryReader().load_data(file=source, extra_info=metadata)
                    else: # Assume text or other flat file, use SimpleDirectoryReader for one file
                        return SimpleDirectoryReader(input_files=[source], file_metadata=lambda _: metadata or {}).load_data()
                elif os.path.isdir(source):
                    return SimpleDirectoryReader(input_dir=source, recursive=True, file_metadata=lambda _: metadata or {}).load_data()
                else:
                    raise DocumentProcessingError(f"Unsupported file/path source: {source}")
            else:
                # Could be a file-like object from an upload
                # This part needs careful handling depending on how FastAPI passes the file
                # For now, assume 'source' is a file path string for simplicity
                raise DocumentProcessingError("Source type not supported. Expected file path string.")
            
            # You might want to enrich documents with metadata here if not done by reader
            # for doc in documents:
            #     doc.metadata = {**(doc.metadata or {}), **(metadata or {})}
            # return documents

        except Exception as e:
            logger.error(f"Error loading document with LlamaIndex: {e}", exc_info=True)
            raise DocumentProcessingError(f"LlamaIndex loader failed: {str(e)}")