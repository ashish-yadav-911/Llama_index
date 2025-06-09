# AGENTIC_MIRAI/app/factory/component_factory.py
from shared.config import Settings
from shared.exceptions import ConfigurationError
from shared.log import get_logger

# Abstract types
from app.abstract.loader import BaseDocumentLoader
from app.abstract.splitter import BaseTextSplitter
from app.abstract.embedder import BaseEmbeddingModel
from app.abstract.vectorstore import BaseVectorStore
from app.abstract.retriever import BaseVectorRetriever

# LlamaIndex Implementations
from app.frameworks.llama_index.loader import LlamaIndexDocumentLoader
from app.frameworks.llama_index.splitter import LlamaIndexTextSplitter
from app.frameworks.llama_index.embedder import LlamaIndexEmbeddingModel
from app.frameworks.llama_index.vectorstore import LlamaIndexVectorStore
from app.frameworks.llama_index.retrievers import LlamaIndexVectorRetriever

#llm-synthesizer imports
from app.abstract.synthesizer import BaseResponseSynthesizer
from app.frameworks.llama_index.llms import get_llama_index_llm # New import
from app.frameworks.llama_index.synthesizers import LlamaIndexResponseSynthesizer # New import
from llama_index.core.llms import LLM # For type hint
# Logging setup

logger = get_logger(__name__)

class ComponentFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Current framework - could be configurable if you truly mix frameworks at runtime
        # For now, hardcoded to LlamaIndex as per primary request
        self.framework = "llama_index" 

    def get_document_loader(self) -> BaseDocumentLoader:
        logger.debug(f"Creating document loader for framework: {self.framework}")
        if self.framework == "llama_index":
            return LlamaIndexDocumentLoader()
        # elif self.framework == "langchain":
        #     from app.frameworks.langchain.loaders import LangChainDocumentLoader # Example
        #     return LangChainDocumentLoader()
        raise ConfigurationError(f"Unsupported framework for document loader: {self.framework}")

    def get_text_splitter(self) -> BaseTextSplitter:
        logger.debug(f"Creating text splitter for framework: {self.framework}")
        if self.framework == "llama_index":
            # LlamaIndexTextSplitter does not take settings at init currently
            return LlamaIndexTextSplitter() 
        raise ConfigurationError(f"Unsupported framework for text splitter: {self.framework}")

    def get_embedding_model(self) -> BaseEmbeddingModel:
        logger.debug(f"Creating embedding model for framework: {self.framework}")
        if self.framework == "llama_index":
            return LlamaIndexEmbeddingModel(settings=self.settings)
        raise ConfigurationError(f"Unsupported framework for embedding model: {self.framework}")

    def get_vector_store(self) -> BaseVectorStore:
        logger.debug(f"Creating vector store for framework: {self.framework}")
        if self.framework == "llama_index":
            return LlamaIndexVectorStore(settings=self.settings)
        raise ConfigurationError(f"Unsupported framework for vector store: {self.framework}")

    def get_vector_retriever(self) -> BaseVectorRetriever:
        logger.debug(f"Creating vector retriever for framework: {self.framework}")
        if self.framework == "llama_index":
            return LlamaIndexVectorRetriever(settings=self.settings)
        raise ConfigurationError(f"Unsupported framework for vector retriever: {self.framework}")
    
    def get_llm(self) -> LLM: # New method
        logger.debug(f"Creating LLM for framework: {self.framework}")
        if self.framework == "llama_index":
            return get_llama_index_llm(settings=self.settings)
        # Add other frameworks later
        raise ConfigurationError(f"Unsupported framework for LLM: {self.framework}")

    def get_response_synthesizer(self) -> BaseResponseSynthesizer:
        logger.debug(f"Creating response synthesizer for framework: {self.framework}")
        if self.framework == "llama_index":
            llm_instance = self.get_llm()
            return LlamaIndexResponseSynthesizer(llm=llm_instance, settings=self.settings) # Call looks correct
        raise ConfigurationError(f"Unsupported framework for response synthesizer: {self.framework}")