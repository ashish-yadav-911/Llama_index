# AGENTIC_MIRAI/app/abstract/splitter.py
from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.core.node_parser import NodeParser # LlamaIndex specific type
from llama_index.core.schema import Document, BaseNode


class BaseTextSplitter(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[BaseNode]:
        """Splits documents into text chunks/nodes."""
        pass

    @abstractmethod
    def get_node_parser(self, chunk_size: int, chunk_overlap: int) -> NodeParser:
        """Returns a framework-specific node parser."""
        pass