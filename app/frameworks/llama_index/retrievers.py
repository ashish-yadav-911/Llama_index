# AGENTIC_MIRAI/app/frameworks/llama_index/retrievers.py
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever # VectorIndexRetriever is a type of BaseRetriever
# from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters # For potential filtering

from app.abstract.retriever import BaseVectorRetriever
from shared.validation.query_schema import QueryRequest
from shared.config import Settings as AppSettings # Your app's settings
from shared.exceptions import UnsupportedFeatureError
from shared.log import get_logger

# LlamaSettings might not be needed here if index.as_retriever handles it all
# from llama_index.core import Settings as LlamaSettings 

logger = get_logger(__name__)

class LlamaIndexVectorRetriever(BaseVectorRetriever):
    def __init__(self, settings: AppSettings): # AppSettings for config like default top_k
        self.app_settings = settings

    def get_retriever(
        self, 
        index: BaseIndex, 
        request_params: QueryRequest
    ) -> BaseRetriever:
        
        top_k = request_params.top_k or self.app_settings.DEFAULT_RETRIEVER_TOP_K
        strategy = request_params.retrieval_strategy or self.app_settings.DEFAULT_RETRIEVER_STRATEGY
        
        logger.info(f"Initializing LlamaIndex retriever with strategy: {strategy}, top_k: {top_k}")

        if strategy == "similarity":
            return index.as_retriever(similarity_top_k=top_k)
        
        elif strategy == "mmr":
            # For MMR, LlamaIndex's as_retriever can take vector_store_query_mode
            # However, MMR parameters like diversity_bias might need to be passed differently
            # or handled via a postprocessor if as_retriever doesn't support it directly.
            # Check LlamaIndex documentation for current `as_retriever` MMR options.
            # mmr_threshold = request_params.mmr_diversity_bias or self.app_settings.MMR_DIVERSITY_BIAS
            try:
                # This is how you'd typically enable MMR if supported directly by the index type
                return index.as_retriever(
                    similarity_top_k=top_k * 2, # Retrieve more candidates for MMR
                    vector_store_query_mode="mmr", 
                    # mmr_prefetch_factor: Optional[float] = None, (some indexes might have this)
                    # mmr_relevance_weight: Optional[float] = None, (check specific index type)
                    # mmr_diversity_bias: Optional[float] = mmr_threshold # Check if this param exists
                )
                # Note: The exact parameters for MMR in as_retriever can vary based on the index type
                # and LlamaIndex version. If `mmr_diversity_bias` isn't a direct param,
                # you might need a NodePostprocessor for MMR.
                logger.info("MMR strategy selected. Configured via index.as_retriever with vector_store_query_mode='mmr'.")
            except TypeError as e:
                logger.warning(f"Could not set MMR mode directly on as_retriever (possibly unsupported params for this index type or LlamaIndex version: {e}). Falling back to similarity retriever. Consider MMRNodePostprocessor.")
                return index.as_retriever(similarity_top_k=top_k) # Fallback
            except Exception as e: # Catch other potential errors
                logger.error(f"Error setting up MMR retriever: {e}. Falling back to similarity.", exc_info=True)
                return index.as_retriever(similarity_top_k=top_k) # Fallback

        else:
            logger.error(f"Unsupported retrieval strategy: {strategy}")
            raise UnsupportedFeatureError(f"Retrieval strategy '{strategy}' not supported.")