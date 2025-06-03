# AGENTIC_MIRAI/app/frameworks/llama_index/splitters.py
from typing import List
from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import Document, BaseNode
from app.abstract.splitter import BaseTextSplitter
from shared.log import get_logger

logger = get_logger(__name__)

class LlamaIndexTextSplitter(BaseTextSplitter):
    def get_node_parser(self, chunk_size: int, chunk_overlap: int) -> NodeParser:
        logger.info(f"Initializing LlamaIndex SentenceSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[BaseNode]:
        parser = self.get_node_parser(chunk_size, chunk_overlap)
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(nodes)} nodes.")
        return nodes