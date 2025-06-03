# AGENTIC_MIRAI/app/abstract/retriever.py
from abc import ABC, abstractmethod
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever
from shared.validation.query_schema import QueryRequest # Using our schema

class BaseVectorRetriever(ABC):
    @abstractmethod
    def get_retriever(
        self, 
        index: BaseIndex, 
        request_params: QueryRequest # Pass the request to allow strategy configuration
    ) -> BaseRetriever:
        """Returns a framework-specific retriever."""
        pass