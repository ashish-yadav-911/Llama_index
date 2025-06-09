# AGENTIC_MIRAI/app/frameworks/llama_index/embedders.py
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from app.abstract.embedder import BaseEmbeddingModel
from shared.config import Settings
from shared.exceptions import ConfigurationError
from shared.log import get_logger

logger = get_logger(__name__)

class LlamaIndexEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_embedding_model(self) -> BaseEmbedding:
        model_type = self.settings.EMBEDDING_MODEL_TYPE
        logger.info(f"Initializing LlamaIndex embedding model: {model_type}")
        if model_type == "huggingface":
            return HuggingFaceEmbedding(model=self.settings.HF_EMBEDDING_MODEL_NAME)
        elif model_type == "openai":
            if not self.settings.OPENAI_API_KEY:
                raise ConfigurationError("OPENAI_API_KEY is not set for OpenAI embeddings.")
            return OpenAIEmbedding(
                    model=self.settings.OPENAI_EMBEDDING_MODEL_NAME,
                api_key=self.settings.OPENAI_API_KEY
            )
        else:
            logger.error(f"Unsupported embedding model type: {model_type}")
            raise ConfigurationError(f"Unsupported embedding model type: {model_type}")