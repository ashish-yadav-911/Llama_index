# AGENTIC_MIRAI/app/abstract/vector_store.py
from abc import ABC, abstractmethod
from llama_index.core.vector_stores.types import VectorStore # LlamaIndex specific type
from llama_index.core.storage.storage_context import StorageContext

class BaseVectorStore(ABC):
    @abstractmethod
    def get_vector_store(self) -> VectorStore:
        """Returns a framework-specific vector store instance."""
        pass
    
    @abstractmethod
    def get_storage_context(self) -> StorageContext:
        """Returns a framework-specific storage context, potentially containing the vector store."""
        pass