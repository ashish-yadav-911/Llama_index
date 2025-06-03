# AGENTIC_MIRAI/app/abstract/embedder.py
from abc import ABC, abstractmethod
from llama_index.core.embeddings import BaseEmbedding # LlamaIndex specific type

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def get_embedding_model(self) -> BaseEmbedding:
        """Returns a framework-specific embedding model instance."""
        pass