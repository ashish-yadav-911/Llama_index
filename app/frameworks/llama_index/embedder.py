# AGENTIC_MIRAI/app/frameworks/llama_index/embedders.py
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from app.abstract.embedder import BaseEmbeddingModel
from shared.config import Settings
from shared.exceptions import ConfigurationError
from shared.log import get_logger

# logger = get_logger(__name__)

# class LlamaIndexEmbeddingModel(BaseEmbeddingModel):
#     def __init__(self, settings: Settings):
#         self.settings = settings

#     def get_embedding_model(self) -> BaseEmbedding:
#         model_type = self.settings.EMBEDDING_MODEL_TYPE
#         logger.info(f"Initializing LlamaIndex embedding model: {model_type}")
#         if model_type == "huggingface":
#             return HuggingFaceEmbedding(model=self.settings.HF_EMBEDDING_MODEL_NAME)
#         elif model_type == "openai":
#             if not self.settings.OPENAI_API_KEY:
#                 raise ConfigurationError("OPENAI_API_KEY is not set for OpenAI embeddings.")
#             return OpenAIEmbedding(
#                     model=self.settings.OPENAI_EMBEDDING_MODEL_NAME,
#                 api_key=self.settings.OPENAI_API_KEY
#             )
#         else:
#             logger.error(f"Unsupported embedding model type: {model_type}")
#             raise ConfigurationError(f"Unsupported embedding model type: {model_type}")










# AGENTIC_MIRAI/app/frameworks/llama_index/embedders.py
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding # Direct import
# from app.abstract.embedder import BaseEmbeddingModel
# from shared.config import Settings as AppSettings
# from shared.exceptions import ConfigurationError
# from shared.log import get_logger

logger = get_logger(__name__)

class LlamaIndexEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, settings: Settings): 
        self.settings = settings

    def get_embedding_model(self) -> BaseEmbedding:
        model_type = self.settings.EMBEDDING_MODEL_TYPE
        logger.info(f"Initializing LlamaIndex embedding model provider: {model_type}")

        if model_type == "huggingface":
            logger.info(f"Using HuggingFace embedding model: {self.settings.HF_EMBEDDING_MODEL_NAME}")
            return HuggingFaceEmbedding(model_name=self.settings.HF_EMBEDDING_MODEL_NAME)
        
        elif model_type == "openai":
            if not self.settings.OPENAI_API_KEY:
                raise ConfigurationError("OPENAI_API_KEY is not set for OpenAI embeddings.")
            
            model_name = self.settings.OPENAI_EMBEDDING_MODEL_NAME
            logger.info(f"Using OpenAI embedding model: {model_name}")
            
            try:
                # Basic initialization for LlamaIndex 0.10.x OpenAIEmbedding
                # It should automatically handle its dimension and other internals based on the model name.
                embedding_instance = OpenAIEmbedding(
                    model=model_name,
                    api_key=self.settings.OPENAI_API_KEY
                    # No need to manually set embed_dim here; let LlamaIndex handle it.
                    # If it can't, the error should be more direct about dimension issues.
                )
                # Verify embed_dim is populated (LlamaIndex 0.10.x should do this for OpenAI)
                if not hasattr(embedding_instance, 'embed_dim') or not embedding_instance.embed_dim:
                    logger.warning(f"LlamaIndex OpenAIEmbedding for model '{model_name}' did not auto-populate 'embed_dim'. "
                                   "This might lead to issues if dimension cannot be inferred later.")
                else:
                    logger.info(f"LlamaIndex OpenAIEmbedding initialized with model '{model_name}', embed_dim: {embedding_instance.embed_dim}")

                return embedding_instance

            except Exception as e:
                logger.error(f"Error initializing OpenAIEmbedding with model '{model_name}': {e}", exc_info=True)
                raise ConfigurationError(f"Failed to initialize OpenAIEmbedding: {e}")
        
        else:
            logger.error(f"Unsupported embedding model type: {model_type}")
            raise ConfigurationError(f"Unsupported embedding model type: {model_type}")