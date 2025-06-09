# # AGENTIC_MIRAI/app/frameworks/llama_index/retrievers.py
# from llama_index.core.indices.base import BaseIndex
# from llama_index.core.retrievers import BaseRetriever # VectorIndexRetriever is a type of BaseRetriever
# # from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters # For potential filtering

# from app.abstract.retriever import BaseVectorRetriever
# from shared.validation.query_schema import QueryRequest
# from shared.config import Settings as AppSettings # Your app's settings
# from shared.exceptions import UnsupportedFeatureError
# from shared.log import get_logger

# # LlamaSettings might not be needed here if index.as_retriever handles it all
# # from llama_index.core import Settings as LlamaSettings 

# logger = get_logger(__name__)

# class LlamaIndexVectorRetriever(BaseVectorRetriever):
#     def __init__(self, settings: AppSettings): # AppSettings for config like default top_k
#         self.app_settings = settings

#     def get_retriever(
#         self, 
#         index: BaseIndex, 
#         request_params: QueryRequest
#     ) -> BaseRetriever:
        
#         top_k = request_params.top_k or self.app_settings.DEFAULT_RETRIEVER_TOP_K
#         strategy = request_params.retrieval_strategy or self.app_settings.DEFAULT_RETRIEVER_STRATEGY
        
#         logger.info(f"Initializing LlamaIndex retriever with strategy: {strategy}, top_k: {top_k}")

#         if strategy == "similarity":
#             return index.as_retriever(similarity_top_k=top_k)
        
#         elif strategy == "mmr":
#             # For MMR, LlamaIndex's as_retriever can take vector_store_query_mode
#             # However, MMR parameters like diversity_bias might need to be passed differently
#             # or handled via a postprocessor if as_retriever doesn't support it directly.
#             # Check LlamaIndex documentation for current `as_retriever` MMR options.
#             # mmr_threshold = request_params.mmr_diversity_bias or self.app_settings.MMR_DIVERSITY_BIAS
#             try:
#                 # This is how you'd typically enable MMR if supported directly by the index type
#                 return index.as_retriever(
#                     similarity_top_k=top_k * 2, # Retrieve more candidates for MMR
#                     vector_store_query_mode="mmr", 
#                     # mmr_prefetch_factor: Optional[float] = None, (some indexes might have this)
#                     # mmr_relevance_weight: Optional[float] = None, (check specific index type)
#                     # mmr_diversity_bias: Optional[float] = mmr_threshold # Check if this param exists
#                 )
#                 # Note: The exact parameters for MMR in as_retriever can vary based on the index type
#                 # and LlamaIndex version. If `mmr_diversity_bias` isn't a direct param,
#                 # you might need a NodePostprocessor for MMR.
#                 logger.info("MMR strategy selected. Configured via index.as_retriever with vector_store_query_mode='mmr'.")
#             except TypeError as e:
#                 logger.warning(f"Could not set MMR mode directly on as_retriever (possibly unsupported params for this index type or LlamaIndex version: {e}). Falling back to similarity retriever. Consider MMRNodePostprocessor.")
#                 return index.as_retriever(similarity_top_k=top_k) # Fallback
#             except Exception as e: # Catch other potential errors
#                 logger.error(f"Error setting up MMR retriever: {e}. Falling back to similarity.", exc_info=True)
#                 return index.as_retriever(similarity_top_k=top_k) # Fallback

#         else:
#             logger.error(f"Unsupported retrieval strategy: {strategy}")
#             raise UnsupportedFeatureError(f"Retrieval strategy '{strategy}' not supported.")


# # AGENTIC_MIRAI/app/frameworks/llama_index/retrievers.py
# from llama_index.core.indices.base import BaseIndex
# from llama_index.core.retrievers import (
#     BaseRetriever, 
#     VectorIndexRetriever, 
#     QueryFusionRetriever
# )
# from llama_index.core.vector_stores.types import VectorStoreQueryMode # For MMR mode constant
# from llama_index.retrievers.bm25 import BM25Retriever # For sparse part of hybrid

# from app.abstract.retriever import BaseVectorRetriever
# from shared.validation.query_schema import QueryRequest
# from shared.config import Settings
# from shared.exceptions import UnsupportedFeatureError, ConfigurationError
# from shared.log import get_logger

# logger = get_logger(__name__)

# class LlamaIndexVectorRetriever(BaseVectorRetriever):
#     def __init__(self, settings: Settings):
#         self.settings = settings

#     def get_retriever(
#         self, 
#         index: BaseIndex, 
#         request_params: QueryRequest
#     ) -> BaseRetriever:
        
#         top_k = request_params.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K
#         strategy = request_params.retrieval_strategy or self.settings.DEFAULT_RETRIEVER_STRATEGY
            
#         logger.info(f"Initializing LlamaIndex retriever with strategy: {strategy}, top_k: {top_k}")

#         if strategy == "similarity":
#             return index.as_retriever(similarity_top_k=top_k)
        
#         elif strategy == "mmr":
#             effective_mmr_threshold = request_params.mmr_threshold if request_params.mmr_threshold is not None else self.settings.DEFAULT_MMR_THRESHOLD
#             logger.info(f"Using MMR strategy with similarity_top_k={top_k}, mmr_threshold={effective_mmr_threshold}")
#             # Note: For MMR, similarity_top_k acts as the number of candidates to fetch before applying MMR.
#             # The final number of results will be <= similarity_top_k, influenced by the threshold.
#             # Some MMR implementations might use a different parameter for final k.
#             # LlamaIndex's as_retriever with MMR mode should respect similarity_top_k as the final desired count after MMR.
#             # However, to be safe, one might fetch more (e.g., top_k * 2) and let MMR refine.
#             # Let's assume similarity_top_k is the target count for now.
#             return index.as_retriever(
#                 vector_store_query_mode=VectorStoreQueryMode.MMR,
#                 similarity_top_k=top_k, 
#                 mmr_threshold=effective_mmr_threshold
#             )
        
#         elif strategy == "hybrid":
#             logger.info(f"Using Hybrid strategy with final top_k={top_k}")
#             try:
#                 # 1. Dense Retriever (standard vector search)
#                 # We might want to fetch more candidates for each part of hybrid search
#                 # dense_top_k = request_params.hybrid_dense_top_k or top_k * 2 # Example
#                 dense_retriever = index.as_retriever(similarity_top_k=top_k) # Fetches 'top_k' dense results

#                 # 2. Sparse Retriever (BM25)
#                 # BM25Retriever needs nodes. This works well if index.docstore has all nodes.
#                 if not hasattr(index, 'docstore') or not index.docstore:
#                     raise ConfigurationError("Hybrid search with BM25 requires an index with an accessible docstore (e.g., from SimpleVectorStore).")
                
#                 nodes = list(index.docstore.docs.values())
#                 if not nodes:
#                     logger.warning("No nodes found in docstore for BM25Retriever. Hybrid search might be ineffective.")
#                     # Fallback to dense retriever only if no nodes for BM25
#                     return dense_retriever 
                
#                 # sparse_top_k = request_params.hybrid_sparse_top_k or top_k * 2 # Example
#                 sparse_retriever = BM25Retriever.from_defaults(
#                     nodes=nodes, 
#                     similarity_top_k=top_k # Fetches 'top_k' sparse results
#                 )

#                 # 3. QueryFusionRetriever to combine results
#                 # mode="RRF" (Reciprocal Rank Fusion) is a common and effective fusion method.
#                 # similarity_top_k for QueryFusionRetriever is the *final* number of results after fusion.
#                 fusion_retriever = QueryFusionRetriever(
#                     retrievers=[dense_retriever, sparse_retriever],
#                     similarity_top_k=top_k, # Final number of results after fusion
#                     num_queries=1,  # Number of search queries to generate (1 for RRF usually)
#                     mode="RRF",
#                     # query_gen_prompt="...",  # Can customize prompt for query generation if num_queries > 1
#                     # use_async=True, # if your retrievers support async
#                 )
#                 logger.info(f"Initialized QueryFusionRetriever for hybrid search (Dense + BM25) with RRF mode.")
#                 return fusion_retriever
            
#             except ImportError:
#                 logger.error("BM25Retriever or its dependency 'rank_bm25' not installed. Hybrid search with BM25 cannot be used.")
#                 raise UnsupportedFeatureError("Hybrid search (BM25) requires 'rank_bm25'. Please install it.")
#             except Exception as e:
#                 logger.error(f"Error setting up hybrid retriever: {e}", exc_info=True)
#                 raise ConfigurationError(f"Could not set up hybrid retriever: {str(e)}")

#         else:
#             logger.error(f"Unsupported retrieval strategy: {strategy}")
#             raise UnsupportedFeatureError(f"Retrieval strategy '{strategy}' not supported.")

     #|^ error in hybrid retriever setup



# AGENTIC_MIRAI/app/frameworks/llama_index/retrievers.py

from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    QueryFusionRetriever,
)
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES 
from llama_index.core.vector_stores.types import VectorStoreQueryMode  # For MMR mode
from llama_index.retrievers.bm25 import BM25Retriever  # For sparse part of hybrid

from app.abstract.retriever import BaseVectorRetriever
from shared.validation.query_schema import QueryRequest
from shared.config import Settings
from shared.exceptions import UnsupportedFeatureError, ConfigurationError
from shared.log import get_logger

logger = get_logger(__name__)


class LlamaIndexVectorRetriever(BaseVectorRetriever):
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_retriever(
        self,
        index: BaseIndex,
        request_params: QueryRequest
    ) -> BaseRetriever:

        top_k = request_params.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K
        strategy = request_params.retrieval_strategy or self.settings.DEFAULT_RETRIEVER_STRATEGY

        logger.info(f"Initializing LlamaIndex retriever with strategy: {strategy}, top_k: {top_k}")

        if strategy == "similarity":
            return index.as_retriever(similarity_top_k=top_k)

        elif strategy == "mmr":
            effective_mmr_threshold = (
                request_params.mmr_threshold
                if request_params.mmr_threshold is not None
                else self.settings.DEFAULT_MMR_THRESHOLD
            )
            logger.info(
                f"Using MMR strategy with similarity_top_k={top_k}, "
                f"mmr_threshold={effective_mmr_threshold}"
            )
            return index.as_retriever(
                vector_store_query_mode=VectorStoreQueryMode.MMR,
                similarity_top_k=top_k,
                mmr_threshold=effective_mmr_threshold
            )

        elif strategy == "hybrid":
            logger.info(f"Using Hybrid strategy with final top_k={top_k}")
            try:
                dense_retriever = index.as_retriever(similarity_top_k=top_k)

                if not hasattr(index, 'docstore') or not index.docstore:
                    raise ConfigurationError(
                        "Hybrid search requires an index with an accessible docstore."
                    )
                nodes = list(index.docstore.docs.values())
                if not nodes:
                    logger.warning(
                        "No nodes found in docstore for BM25Retriever. "
                        "Falling back to dense retriever."
                    )
                    return dense_retriever

                sparse_retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=top_k
                )

                fusion_retriever = QueryFusionRetriever(
                    retrievers=[dense_retriever, sparse_retriever],
                    similarity_top_k=top_k,
                    num_queries=1,
                    mode=FUSION_MODES.RECIPROCAL_RANK  # ✅ Correct enum usage
                )

                logger.info("Initialized QueryFusionRetriever with RECIPROCAL_RANK mode.")
                return fusion_retriever

            except ImportError:
                logger.error(
                    "BM25Retriever or 'rank_bm25' not installed. "
                    "Install dependencies for hybrid support."
                )
                raise UnsupportedFeatureError(
                    "Hybrid search (BM25) requires 'rank_bm25'. Please install it."
                )
            except Exception as e:
                logger.error(f"Error setting up hybrid retriever: {e}", exc_info=True)
                raise ConfigurationError(f"Could not set up hybrid retriever: {str(e)}")

        else:
            logger.error(f"Unsupported retrieval strategy: {strategy}")
            raise UnsupportedFeatureError(f"Retrieval strategy '{strategy}' not supported.")
