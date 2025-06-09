# # AGENTIC_MIRAI/app/frameworks/llama_index/vectorstore.py
# from typing import Optional

# from llama_index.core.vector_stores.types import VectorStore
# from llama_index.vector_stores.pinecone import PineconeVectorStore as LlamaPineconeVectorStore 
# from llama_index.core.storage.storage_context import StorageContext
# from llama_index.core.settings import Settings as LlamaSettings 

# from app.abstract.vectorstore import BaseVectorStore
# from shared.config import Settings as AppSettings 
# from shared.exceptions import ConfigurationError
# from shared.log import get_logger

# # Correct imports for the NEW 'pinecone' package (e.g., v3.x, v4.x)
# from pinecone import Pinecone # Use the main class directly
# # For exceptions, based on recent Pinecone SDK versions:
# from pinecone.core.client.exceptions import PineconeApiException, NotFoundException # Import specific exceptions you might handle

# logger = get_logger(__name__)

# class LlamaIndexVectorStore(BaseVectorStore):
#     def __init__(self, settings: AppSettings):
#         self.settings = settings
#         self._vector_store_instance: Optional[VectorStore] = None

#     def get_vector_store(self) -> VectorStore:
#         if self._vector_store_instance:
#             return self._vector_store_instance

#         store_type = self.settings.VECTOR_STORE_TYPE
#         logger.info(f"Initializing LlamaIndex vector store: {store_type}")

#         if store_type == "simple":
#             from llama_index.core.vector_stores.simple import SimpleVectorStore
#             self._vector_store_instance = SimpleVectorStore()
        
#         elif store_type == "pinecone":
#             if not self.settings.PINECONE_API_KEY:
#                 raise ConfigurationError("Pinecone API key (PINECONE_API_KEY) is not set.")
#             if not self.settings.PINECONE_INDEX_HOST: # Host is essential
#                 raise ConfigurationError("Pinecone Index Host (PINECONE_INDEX_HOST) must be set.")

#             log_conn_details = f"index host: '{self.settings.PINECONE_INDEX_HOST}'"
#             if self.settings.PINECONE_NAMESPACE:
#                 log_conn_details += f", namespace: '{self.settings.PINECONE_NAMESPACE}'"
            
#             logger.info(f"Attempting to connect to Pinecone with {log_conn_details}")

#             try:
#                 pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
#                 logger.info(f"Using Pinecone index host: {self.settings.PINECONE_INDEX_HOST}")
#                 pinecone_index_handle = pc.Index(host=self.settings.PINECONE_INDEX_HOST)
                
#                 pinecone_actual_dimension = None
#                 try:
#                     stats = pinecone_index_handle.describe_index_stats()
#                     logger.info(f"Successfully connected to Pinecone index. Stats: {stats}")
#                     pinecone_actual_dimension = stats.dimension # Get actual dimension from Pinecone
#                 except NotFoundException:
#                     logger.error(f"Pinecone index at host '{self.settings.PINECONE_INDEX_HOST}' not found.")
#                     raise ConfigurationError(f"Pinecone index at host '{self.settings.PINECONE_INDEX_HOST}' not found.")
#                 except PineconeApiException as e_stats:
#                     logger.warning(f"Could not get stats from Pinecone index, but handle obtained. Error: {e_stats}")
                
#                 # Dimension determination for LlamaIndex embedding model
#                 if not LlamaSettings.embed_model:
#                     raise ConfigurationError("Embedding model not set in LlamaSettings; cannot verify/determine dimension for Pinecone.")

#                 llama_model_dimension = None
#                 # Try to get embed_dim attribute directly first
#                 if hasattr(LlamaSettings.embed_model, 'embed_dim') and LlamaSettings.embed_model.embed_dim:
#                     llama_model_dimension = LlamaSettings.embed_model.embed_dim
#                     logger.info(f"Retrieved 'embed_dim' directly from LlamaSettings.embed_model: {llama_model_dimension}")
#                 else:
#                     # Fallback: Try to get a dummy embedding if embed_dim is not available/set
#                     logger.info("Attempting to determine embedding dimension via dummy embedding as 'embed_dim' attribute was not found or was None.")
#                     try:
#                         dummy_embedding = LlamaSettings.embed_model.get_text_embedding("hello")
#                         llama_model_dimension = len(dummy_embedding)
#                         logger.info(f"Determined embedding dimension via dummy embedding: {llama_model_dimension}")
#                     except AttributeError as e_attr: # Catch specific AttributeError if internal attributes are missing
#                         logger.error(f"AttributeError while getting dummy embedding (model: {type(LlamaSettings.embed_model).__name__}): {e_attr}", exc_info=True)
#                         raise ConfigurationError(f"Could not determine embedding dimension due to AttributeError in embedding model: {e_attr}")
#                     except Exception as e_dim: # Catch other errors during dummy embedding
#                         logger.error(f"Generic error while getting dummy embedding: {e_dim}", exc_info=True)
#                         raise ConfigurationError(f"Could not determine embedding dimension via dummy embedding: {e_dim}")
                
#                 if not llama_model_dimension: # Should be caught by exceptions above, but as a safeguard
#                     raise ConfigurationError("Failed to obtain embedding dimension from LlamaSettings.embed_model.")

#                 # Compare with Pinecone's actual dimension if available
#                 if pinecone_actual_dimension and pinecone_actual_dimension != llama_model_dimension:
#                     raise ConfigurationError(
#                         f"Dimension mismatch! Pinecone index ('{self.settings.INDEX_NAME}') has dimension {pinecone_actual_dimension}, "
#                         f"but LlamaIndex embedding model ('{self.settings.OPENAI_EMBEDDING_MODEL_NAME}') implies dimension {llama_model_dimension}."
#                     )
#                 elif pinecone_actual_dimension: # Dimensions match or Pinecone dimension was not fetched (due to warning)
#                     logger.info(f"Pinecone index dimension ({pinecone_actual_dimension}) and LlamaIndex model dimension ({llama_model_dimension}) are consistent or Pinecone dim not checked.")
#                 else: # Pinecone dimension not fetched, proceed with LlamaIndex model dimension
#                     logger.warning(f"Could not fetch Pinecone index dimension. Proceeding with LlamaIndex model dimension: {llama_model_dimension}")


#                 self._vector_store_instance = LlamaPineconeVectorStore(
#                     pinecone_index=pinecone_index_handle,
#                     namespace=self.settings.PINECONE_NAMESPACE or None,
#                     text_key="chunk_text" 
#                 )
#                 logger.info("Successfully initialized LlamaIndex PineconeVectorStore.")

#             except PineconeApiException as e: # Catching the specific, correctly imported exception
#                 logger.error(f"Pinecone API Exception: {e}", exc_info=True)
#                 error_body = str(e) # Default to string representation
#                 if hasattr(e, 'body') and e.body: error_body = str(e.body)
#                 status_code = e.status if hasattr(e, 'status') else "N/A"
#                 raise ConfigurationError(f"Pinecone API error (status: {status_code}): {error_body}")
#             except Exception as e:
#                 logger.error(f"Failed to initialize PineconeVectorStore: {e}", exc_info=True)
#                 raise ConfigurationError(f"Error initializing Pinecone: {str(e)}")

#         elif store_type == "postgres":
#             # ... (postgres logic) ...
#             pass # Placeholder for brevity
#         else:
#             logger.error(f"Unsupported vector store type: {store_type}")
#             raise ConfigurationError(f"Unsupported vector store type: {store_type}")
        
#         return self._vector_store_instance

#     def get_storage_context(self) -> StorageContext:
#         vector_store = self.get_vector_store()
#         return StorageContext.from_defaults(vector_store=vector_store)









# AGENTIC_MIRAI/app/frameworks/llama_index/vectorstore.py
from typing import Optional

from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore as LlamaPineconeVectorStore 
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.settings import Settings as LlamaSettings 
from llama_index.embeddings.openai import OpenAIEmbedding # Import to check instance type

from app.abstract.vectorstore import BaseVectorStore
from shared.config import Settings as AppSettings 
from shared.exceptions import ConfigurationError
from shared.log import get_logger

from pinecone import Pinecone 
from pinecone.core.client.exceptions import PineconeApiException, NotFoundException 

logger = get_logger(__name__)

class LlamaIndexVectorStore(BaseVectorStore):
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._vector_store_instance: Optional[VectorStore] = None

    def get_vector_store(self) -> VectorStore:
        if self._vector_store_instance:
            return self._vector_store_instance

        store_type = self.settings.VECTOR_STORE_TYPE
        logger.info(f"Initializing LlamaIndex vector store: {store_type}")

        if store_type == "simple":
            from llama_index.core.vector_stores.simple import SimpleVectorStore
            self._vector_store_instance = SimpleVectorStore()
        
        elif store_type == "pinecone":
            if not self.settings.PINECONE_API_KEY:
                raise ConfigurationError("Pinecone API key (PINECONE_API_KEY) is not set.")
            if not self.settings.PINECONE_INDEX_HOST:
                raise ConfigurationError("Pinecone Index Host (PINECONE_INDEX_HOST) must be set.")

            log_conn_details = f"index host: '{self.settings.PINECONE_INDEX_HOST}'"
            if self.settings.PINECONE_NAMESPACE:
                log_conn_details += f", namespace: '{self.settings.PINECONE_NAMESPACE}'"
            logger.info(f"Attempting to connect to Pinecone with {log_conn_details}")

            try:
                pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
                pinecone_index_handle = pc.Index(host=self.settings.PINECONE_INDEX_HOST)
                
                pinecone_actual_dimension = None
                try:
                    stats = pinecone_index_handle.describe_index_stats()
                    logger.info(f"Successfully connected to Pinecone index. Stats: {stats}")
                    pinecone_actual_dimension = stats.dimension
                except NotFoundException:
                    logger.error(f"Pinecone index at host '{self.settings.PINECONE_INDEX_HOST}' not found.")
                    raise ConfigurationError(f"Pinecone index at host '{self.settings.PINECONE_INDEX_HOST}' not found.")
                except PineconeApiException as e_stats:
                    logger.warning(f"Could not get stats from Pinecone index. Error: {e_stats}. Will proceed if possible.")
                
                # --- Dimension Determination for LlamaIndex Embedding Model ---
                if not LlamaSettings.embed_model:
                    raise ConfigurationError("Embedding model not set in LlamaSettings; cannot determine dimension.")

                llama_model_dimension = None
                
                # 1. Check if embed_model is OpenAI and if we know its dimension from AppSettings
                if isinstance(LlamaSettings.embed_model, OpenAIEmbedding):
                    model_name_from_config = self.settings.OPENAI_EMBEDDING_MODEL_NAME
                    known_openai_dimensions = {
                        "text-embedding-3-large": 3072,
                        "text-embedding-3-small": 1536,
                        "text-embedding-ada-002": 1536,
                    }
                    if model_name_from_config in known_openai_dimensions:
                        llama_model_dimension = known_openai_dimensions[model_name_from_config]
                        logger.info(f"Using known dimension {llama_model_dimension} for OpenAI model '{model_name_from_config}'.")
                    else:
                        logger.warning(f"OpenAI model '{model_name_from_config}' not in known dimension map. Will try other methods.")

                # 2. If not found above, try to get 'embed_dim' attribute directly from the LlamaIndex embedding object
                if not llama_model_dimension:
                    if hasattr(LlamaSettings.embed_model, 'embed_dim') and LlamaSettings.embed_model.embed_dim:
                        llama_model_dimension = LlamaSettings.embed_model.embed_dim
                        logger.info(f"Using 'embed_dim' ({llama_model_dimension}) directly from LlamaSettings.embed_model ({type(LlamaSettings.embed_model).__name__}).")
                    else:
                        # 3. FALLBACK: Try to get a dummy embedding (THIS IS THE PART THAT WAS FAILING)
                        logger.warning(f"'embed_dim' attribute not found or is None on LlamaSettings.embed_model. "
                                       "Attempting to determine dimension via dummy embedding. This might fail for some models/versions.")
                        try:
                            dummy_embedding = LlamaSettings.embed_model.get_text_embedding("hello")
                            llama_model_dimension = len(dummy_embedding)
                            logger.info(f"Determined embedding dimension ({llama_model_dimension}) via dummy embedding.")
                        except AttributeError as e_attr: 
                            logger.error(f"ATTRIBUTE ERROR while getting dummy embedding (model: {type(LlamaSettings.embed_model).__name__}): {e_attr}", exc_info=True)
                            raise ConfigurationError(f"CRITICAL: Could not determine embedding dimension due to AttributeError. Embedding model seems improperly initialized for this operation. Error: {e_attr}")
                        except Exception as e_dim: 
                            logger.error(f"Generic error while getting dummy embedding: {e_dim}", exc_info=True)
                            raise ConfigurationError(f"CRITICAL: Could not determine embedding dimension via dummy embedding. Error: {e_dim}")
                
                if not llama_model_dimension: 
                    raise ConfigurationError("CRITICAL: Failed to obtain a valid embedding dimension from LlamaSettings.embed_model after all attempts.")
                
                # Compare with Pinecone's actual dimension if it was successfully fetched
                if pinecone_actual_dimension and pinecone_actual_dimension != llama_model_dimension:
                    raise ConfigurationError(
                        f"Dimension mismatch! Pinecone index ('{self.settings.INDEX_NAME}' via host) has dimension {pinecone_actual_dimension}, "
                        f"but LlamaIndex embedding model implies dimension {llama_model_dimension}."
                    )
                elif pinecone_actual_dimension: 
                    logger.info(f"Pinecone index dimension ({pinecone_actual_dimension}) and LlamaIndex model dimension ({llama_model_dimension}) are consistent.")
                else: 
                    logger.warning(f"Could not fetch Pinecone index dimension (stats error). Proceeding with LlamaIndex model dimension: {llama_model_dimension}")

                self._vector_store_instance = LlamaPineconeVectorStore(
                    pinecone_index=pinecone_index_handle,
                    namespace=self.settings.PINECONE_NAMESPACE or None,
                    text_key="chunk_text" 
                )
                logger.info("Successfully initialized LlamaIndex PineconeVectorStore.")

            except PineconeApiException as e: 
                logger.error(f"Pinecone API Exception: {e}", exc_info=True)
                error_body = str(e)
                if hasattr(e, 'body') and e.body: error_body = str(e.body)
                status_code = e.status if hasattr(e, 'status') else "N/A"
                raise ConfigurationError(f"Pinecone API error (status: {status_code}): {error_body}")
            except Exception as e:
                logger.error(f"Failed to initialize PineconeVectorStore: {e}", exc_info=True)
                raise ConfigurationError(f"Error initializing Pinecone: {str(e)}")

        elif store_type == "postgres":
            # ... (Postgres logic remains the same)
            pass # Placeholder
        else:
            logger.error(f"Unsupported vector store type: {store_type}")
            raise ConfigurationError(f"Unsupported vector store type: {store_type}")
        
        return self._vector_store_instance

    def get_storage_context(self) -> StorageContext:
        vector_store = self.get_vector_store()
        return StorageContext.from_defaults(vector_store=vector_store)