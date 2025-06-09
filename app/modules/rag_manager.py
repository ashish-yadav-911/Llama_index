# # AGENTIC_MIRAI/app/modules/rag_manager.py
# from typing import List, Optional, Any, Dict
# from pathlib import Path
# from llama_index.core import load_index_from_storage
# import shutil

# # LlamaIndex core imports
# from llama_index.core import VectorStoreIndex, StorageContext, Document
# from llama_index.core.indices.base import BaseIndex
# from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
# from llama_index.core.node_parser import NodeParser
# from llama_index.core.embeddings import BaseEmbedding
# from llama_index.core.llms import LLM as LlamaLLM # Alias to avoid conflict
# from llama_index.core.query_engine import RetrieverQueryEngine

# # Import global LlamaIndex settings
# from llama_index.core.settings import Settings as LlamaSettings
# # Alternatively, you might see `from llama_index.core import Settings as LlamaSettings`
# # Let's stick to `llama_index.core.settings` as per the migration guide's typical import style.

# from app.factory.component_factory import ComponentFactory
# from shared.config import Settings as AppSettings # Alias our app settings
# from shared.log import get_logger
# from shared.exceptions import DocumentProcessingError, IndexingError, QueryError, ConfigurationError
# from shared.validation.query_schema import QueryRequest, RetrievedNode, DocumentMetadata

# # LlamaIndex specific types from our abstractions (if needed, but mostly using core types now)
# # from app.abstract.vector_store import BaseVectorStore as AbstractVectorStore

# logger = get_logger(__name__)

# class RAGManager:
#     def __init__(self, settings: AppSettings, component_factory: ComponentFactory):
#         self.settings: AppSettings = settings # App-specific settings
#         self.factory = component_factory
#         self._persist_directory_path: Optional[str] = None
#         self._embed_model_instance: Optional[BaseEmbedding] = None
#         self._llm_instance: Optional[LlamaLLM] = None # For LlamaIndex's LLM type
        
#         # Vector store related attributes
#         self._vector_store_manager = None # This is our abstract BaseVectorStore manager
#         self._vector_store_llama_instance = None # This is the actual LlamaIndex VectorStore instance
#         self._storage_context: Optional[StorageContext] = None
        
#         self._index: Optional[BaseIndex] = None # This will hold the LlamaIndex VectorStoreIndex

#         self._initialize_components_and_settings()

#     def _initialize_llm(self):
#         """Initializes the LLM based on configuration and sets it in LlamaSettings."""
#         if self.settings.LLM_PROVIDER:
#             logger.info(f"Initializing LLM for LlamaIndex Settings: Provider {self.settings.LLM_PROVIDER}")
#             # This part would involve a factory method similar to get_embedding_model
#             # For now, let's assume an OpenAI LLM if configured
#             if self.settings.LLM_PROVIDER == "openai":
#                 if not self.settings.OPENAI_API_KEY:
#                     raise ConfigurationError("OPENAI_API_KEY must be set to use OpenAI LLM.")
#                 try:
#                     from llama_index.llms.openai import OpenAI as OpenAI_LLM # Renamed to avoid conflict
#                     self._llm_instance = OpenAI_LLM(
#                         model=self.settings.OPENAI_LLM_MODEL_NAME,
#                         api_key=self.settings.OPENAI_API_KEY
#                     )
#                     LlamaSettings.llm = self._llm_instance
#                     logger.info(f"LlamaIndex global LLM set to: {self.settings.OPENAI_LLM_MODEL_NAME}")
#                 except ImportError:
#                     logger.error("llama-index-llms-openai is not installed. Cannot initialize OpenAI LLM.")
#                     raise ConfigurationError("llama-index-llms-openai is not installed.")
#                 except Exception as e:
#                     logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
#                     raise ConfigurationError(f"Failed to initialize OpenAI LLM: {e}")
#             else:
#                 logger.warning(f"LLM provider '{self.settings.LLM_PROVIDER}' not yet implemented for LlamaSettings in this RAGManager.")
#         else:
#             LlamaSettings.llm = None # Explicitly set to None if no provider
#             logger.info("No LLM provider configured; LlamaIndex global LLM set to None.")


#     def _initialize_components_and_settings(self):
#         logger.info("Initializing RAGManager components and LlamaIndex global settings...")
#         try:
#             # 1. Initialize and set Embedding Model in LlamaSettings
#             embed_model_manager = self.factory.get_embedding_model()
#             self._embed_model_instance = embed_model_manager.get_embedding_model()
#             LlamaSettings.embed_model = self._embed_model_instance # Set globally
#             logger.info(f"LlamaIndex global embedding model initialized: {type(self._embed_model_instance)}")

#             # 2. Initialize and set LLM in LlamaSettings (if configured)
#             self._initialize_llm() # This will set LlamaSettings.llm

#             # 3. Initialize Vector Store components
#             self._vector_store_manager = self.factory.get_vector_store()
#             self._vector_store_llama_instance = self._vector_store_manager.get_vector_store()
            
#             self._persist_directory_path = None # Reset before potentially setting
#             if self.settings.VECTOR_STORE_TYPE == "simple":
#                 self._persist_directory_path = "./storage" # Or make this configurable
#                 logger.info(f"SimpleVectorStore persistence enabled. Target directory: {self._persist_directory_path}")
#                 Path(self._persist_directory_path).mkdir(parents=True, exist_ok=True)

#             self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store_llama_instance)
#             logger.info(f"Vector store initialized: {type(self._vector_store_llama_instance)}")
#             logger.info(f"Storage context configured with vector store: {type(self._storage_context.vector_store)}")

#             # 4. Set global chunking settings (optional, can be overridden per operation)
#             # LlamaSettings.chunk_size = self.settings.DEFAULT_CHUNK_SIZE
#             # LlamaSettings.chunk_overlap = self.settings.DEFAULT_CHUNK_OVERLAP
#             # For node parser, we will create it on-the-fly for ingestion with specific params.
#             # If a global node_parser is desired:
#             # default_node_parser = self.factory.get_text_splitter().get_node_parser(
#             #     chunk_size=self.settings.DEFAULT_CHUNK_SIZE,
#             #     chunk_overlap=self.settings.DEFAULT_CHUNK_OVERLAP
#             # )
#             # LlamaSettings.node_parser = default_node_parser

#             # 5. Load or create the index (relies on global LlamaSettings now)
#             self._load_or_create_index()

#         except Exception as e:
#             logger.error(f"Error during RAGManager component initialization: {e}", exc_info=True)
#             # Reset global settings on failure to avoid partial configurations?
#             # LlamaSettings.llm = None
#             # LlamaSettings.embed_model = None 
#             # LlamaSettings.node_parser = None # etc.
#             raise ConfigurationError(f"Failed to initialize RAG components: {e}")

#     def _load_or_create_index(self):
#         logger.info(f"Attempting to load index '{self.settings.INDEX_NAME}' using vector store '{self.settings.VECTOR_STORE_TYPE}'")
#         try:
#             # VectorStoreIndex will now use LlamaSettings.embed_model, LlamaSettings.llm, etc.
#             if self.settings.VECTOR_STORE_TYPE in ["pinecone", "postgres"]:
#                 self._index = VectorStoreIndex.from_vector_store(
#                     vector_store=self._vector_store_llama_instance,
#                     storage_context=self._storage_context # For from_vector_store, storage_context can be important
#                                                           # to ensure it knows about the store.
#                 )
#                 logger.info(f"Connected to existing vector store '{self.settings.VECTOR_STORE_TYPE}' as index '{self.settings.INDEX_NAME}'.")
            
#             elif self.settings.VECTOR_STORE_TYPE == "simple":
#                 # Check if the configured persist_directory_path exists and has index files
#                 can_load_from_storage = False
#                 if self._persist_directory_path:
#                     p_dir = Path(self._persist_directory_path)
#                     # A more reliable check might be for specific files LlamaIndex creates,
#                     # e.g., docstore.json, vector_store.json, index_store.json
#                     if (p_dir / "docstore.json").exists() and \
#                        (p_dir / "index_store.json").exists() and \
#                        (p_dir / "vector_store.json").exists(): # Basic check
#                         can_load_from_storage = True
                
#                 if can_load_from_storage:
#                     try:
#                         logger.info(f"Attempting to load SimpleVectorStore index from: {self._persist_directory_path}")
#                         # Re-create a storage context pointing to the persist_dir for loading
#                         # The LlamaSettings.embed_model etc. will be used by load_index_from_storage
#                         loading_storage_context = StorageContext.from_defaults(persist_dir=self._persist_directory_path)
#                         self._index = load_index_from_storage(loading_storage_context)
#                         logger.info(f"Successfully loaded SimpleVectorStore index from {self._persist_directory_path}.")
#                         if self._index and self._index.docstore:
#                              logger.info(f"Loaded index has {len(self._index.docstore.docs)} documents.")
#                         else:
#                              logger.warning("Loaded index or its docstore is None/empty after loading from storage.")
#                     except Exception as e:
#                         logger.warning(f"Failed to load index from storage at {self._persist_directory_path}: {e}. Creating new index.", exc_info=True)
#                         self._index = VectorStoreIndex.from_documents(
#                             [],
#                             storage_context=self._storage_context # Use the initially created storage_context
#                         )
#                 else:
#                     logger.info(f"No existing/complete persisted data found for SimpleVectorStore at {self._persist_directory_path or 'default in-memory'}. Creating new index.")
#                     self._index = VectorStoreIndex.from_documents(
#                         [],
#                         storage_context=self._storage_context # Use the initially created storage_context
#                     )
#             else:
#                  raise ConfigurationError(f"Unsupported vector store type for index loading: {self.settings.VECTOR_STORE_TYPE}")

#             if self._index is None: # Fallback, should ideally be handled by above logic
#                  logger.info("Creating a new, empty index as a fallback.")
#                  self._index = VectorStoreIndex.from_documents(
#                     documents=[],
#                     storage_context=self._storage_context
#                 )
            
#             logger.info(f"Index '{self.settings.INDEX_NAME}' is ready.")

#         except Exception as e:
#             logger.error(f"Failed to load or create index: {e}", exc_info=True)
#             raise IndexingError(f"Failed to load or create index '{self.settings.INDEX_NAME}': {e}")


#     def add_document(self, file_path: str, doc_metadata: Optional[Dict[str, Any]] = None,
#                      chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
#         logger.info(f"Processing document: {file_path}")
#         try:
#             loader = self.factory.get_document_loader()
#             splitter_factory = self.factory.get_text_splitter() # This is our abstract splitter

#             effective_chunk_size = chunk_size if chunk_size is not None else self.settings.DEFAULT_CHUNK_SIZE
#             effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.DEFAULT_CHUNK_OVERLAP
            
#             logger.debug(f"Using chunk_size={effective_chunk_size}, chunk_overlap={effective_chunk_overlap}")

#             # 1. Load
#             loaded_docs: List[Document] = loader.load(source=file_path, metadata=doc_metadata)
#             if not loaded_docs:
#                 logger.warning(f"No documents loaded from {file_path}")
#                 return 0
#             logger.info(f"Loaded {len(loaded_docs)} document parts from {file_path}.")

#             # 2. Split (Chunk) using a specific NodeParser instance for this ingestion
#             node_parser: NodeParser = splitter_factory.get_node_parser(
#                 chunk_size=effective_chunk_size, 
#                 chunk_overlap=effective_chunk_overlap
#             )
#             # Manually apply the include/exclude metadata from LlamaSettings if needed,
#             # or ensure NodeParser respects them by default.
#             # node_parser.include_metadata = LlamaSettings.include_metadata 
#             # node_parser.include_prev_next_rel = LlamaSettings.include_prev_next_rel

#             nodes: List[BaseNode] = node_parser.get_nodes_from_documents(loaded_docs)
#             logger.info(f"Generated {len(nodes)} nodes from document with custom chunking.")

#             if not nodes:
#                 logger.warning("No nodes generated after splitting. Nothing to index.")
#                 return 0

#             # 3. Embed & Store (Indexing)
#             # The index.insert_nodes will use LlamaSettings.embed_model
#             self._index.insert_nodes(nodes)
            
#             logger.info(f"Successfully indexed {len(nodes)} nodes from {file_path} into '{self.settings.INDEX_NAME}'.")

#             # Persistence for SimpleVectorStore if storage_context has persist_dir
#             if self.settings.VECTOR_STORE_TYPE == "simple" and self._storage_context.persist_dir:
#                  logger.info(f"Persisting SimpleVectorStore to: {self._storage_context.persist_dir}")
#                  self._index.storage_context.persist(persist_dir=self._storage_context.persist_dir)
#                  # Or just self._storage_context.persist(...) if index has the same storage_context ref

#             return len(nodes)

#         except Exception as e:
#             logger.error(f"Error adding document {file_path}: {e}", exc_info=True)
#             raise IndexingError(f"Failed to process and index document {file_path}: {e}")

#     def query(self, request: QueryRequest) -> List[RetrievedNode]:
#         logger.info(f"Received query: '{request.query_text}' with params: {request.model_dump_json(exclude_none=True)}")
#         if self._index is None:
#             logger.error("Index is not initialized. Cannot perform query.")
#             raise QueryError("Index not initialized. Add documents first.")

#         try:
#             retriever_manager = self.factory.get_vector_retriever()
#             # The retriever is configured based on QueryRequest parameters
#             # The index object is passed, retriever should use LlamaSettings or index's components
#             retriever = retriever_manager.get_retriever(self._index, request)
            
#             logger.debug(f"Using retriever: {type(retriever)}")

#             # Postprocessors can be added here if needed, based on request.retrieval_strategy
#             # For MMR, a true MMR postprocessor would be ideal.
#             # Example:
#             # postprocessors = []
#             # if request.retrieval_strategy == "mmr":
#             #     from llama_index.core.postprocessor import MMRNodePostprocessor # Fictional or from contrib
#             #     mmr_postprocessor = MMRNodePostprocessor(
#             #         top_n=request.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K,
#             #         mmr_threshold= # some value for diversity
#             #     )
#             #     postprocessors.append(mmr_postprocessor)
            
#             # If using postprocessors, QueryEngine might be more convenient
#             # query_engine = self._index.as_query_engine(
#             #    retriever=retriever, 
#             #    node_postprocessors=postprocessors
#             # )
#             # response = query_engine.query(request.query_text)
#             # retrieved_nodes_with_score = response.source_nodes

#             # Direct retrieval:
#             retrieved_nodes_with_score: List[NodeWithScore] = retriever.retrieve(request.query_text)
            
#             # Format response
#             results = []
#             for node_ws in retrieved_nodes_with_score:
#                 doc_meta = DocumentMetadata(
#                     source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
#                     page_number=node_ws.node.metadata.get("page_label"),
#                     extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
#                 )
#                 results.append(RetrievedNode(
#                     text=node_ws.node.get_content(),
#                     score=node_ws.score,
#                     metadata=doc_meta
#                 ))
            
#             logger.info(f"Query '{request.query_text}' retrieved {len(results)} nodes.")
#             return results

#         except Exception as e:
#             logger.error(f"Error during query execution: {e}", exc_info=True)
#             raise QueryError(f"Failed to execute query '{request.query_text}': {e}")
#     def query(self, request: QueryRequest) -> QueryResponse: # Return QueryResponse object
#         logger.info(f"Received query: '{request.query_text}' with params: {request.model_dump_json(exclude_none=True)}")
#         if self._index is None:
#             logger.error("Index is not initialized. Cannot perform query.")
#             raise QueryError("Index not initialized. Add documents first.")
#         if LlamaSettings.llm is None: # Check if an LLM is configured for synthesis
#             logger.warning("No LLM configured in LlamaSettings. Synthesis will be skipped.")
#             # Fallback to retrieval-only behavior
#             retrieved_nodes_obj_list = self._retrieve_only(request)
#             return QueryResponse(
#                 query_text=request.query_text,
#                 retrieved_nodes=retrieved_nodes_obj_list,
#                 synthesized_answer=None
#             )

#         try:
#             retriever_manager = self.factory.get_vector_retriever()
#             retriever = retriever_manager.get_retriever(self._index, request)
            
#             logger.debug(f"Using retriever: {type(retriever)}")

#             # Create a Query Engine. It will use the retriever and LlamaSettings.llm
#             # You can also customize response_synthesizer, node_postprocessors here
#             query_engine = RetrieverQueryEngine.from_args(
#                 retriever=retriever,
#                 # llm=LlamaSettings.llm, # Implicitly uses LlamaSettings.llm
#                 # service_context=self._service_context, # Deprecated
#                 # node_postprocessors=[...], # If you have rerankers/MMR postprocessors
#             )
            
#             logger.info(f"Querying with engine for: '{request.query_text}'")
#             response = query_engine.query(request.query_text) # This is a LlamaIndex Response object

#             # Extract retrieved nodes
#             retrieved_nodes_output: List[RetrievedNode] = []
#             if response.source_nodes:
#                 for node_ws in response.source_nodes:
#                     doc_meta = DocumentMetadata(
#                         source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
#                         page_number=node_ws.node.metadata.get("page_label"),
#                         extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
#                     )
#                     retrieved_nodes_output.append(RetrievedNode(
#                         text=node_ws.node.get_content(),
#                         score=node_ws.score,
#                         metadata=doc_meta
#                     ))
            
#             synthesized_text = str(response) # The string representation of LlamaIndex Response is usually the synthesized answer

#             logger.info(f"Query '{request.query_text}' retrieved {len(retrieved_nodes_output)} source nodes and synthesized answer.")
            
#             return QueryResponse(
#                 query_text=request.query_text,
#                 retrieved_nodes=retrieved_nodes_output,
#                 synthesized_answer=synthesized_text
#             )

#         except Exception as e:
#             logger.error(f"Error during query execution with engine: {e}", exc_info=True)
#             # Fallback to retrieval only if query engine fails for some reason
#             logger.warning("Falling back to retrieval-only due to query engine error.")
#             try:
#                 retrieved_nodes_obj_list_fallback = self._retrieve_only(request)
#                 return QueryResponse(
#                     query_text=request.query_text,
#                     retrieved_nodes=retrieved_nodes_obj_list_fallback,
#                     synthesized_answer=f"Error during synthesis: {e}. Retrieval only."
#                 )
#             except Exception as retrieve_e:
#                 logger.error(f"Error during fallback retrieval: {retrieve_e}", exc_info=True)
#                 raise QueryError(f"Failed to execute query '{request.query_text}' (engine and fallback retrieval failed): {e}")


#     def _retrieve_only(self, request: QueryRequest) -> List[RetrievedNode]:
#         """Helper method for retrieval-only logic, used as fallback or if no LLM."""
#         logger.info(f"Performing retrieval-only for query: '{request.query_text}'")
#         retriever_manager = self.factory.get_vector_retriever()
#         retriever = retriever_manager.get_retriever(self._index, request)
#         retrieved_nodes_with_score: List[NodeWithScore] = retriever.retrieve(request.query_text)
        
#         results = []
#         for node_ws in retrieved_nodes_with_score:
#             doc_meta = DocumentMetadata(
#                 source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
#                 page_number=node_ws.node.metadata.get("page_label"),
#                 extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
#             )
#             results.append(RetrievedNode(
#                 text=node_ws.node.get_content(),
#                 score=node_ws.score,
#                 metadata=doc_meta
#             ))
#         logger.info(f"Retrieval-only for '{request.query_text}' found {len(results)} nodes.")
#         return results

#     def clear_index(self, are_you_sure: bool = False):
#         if not are_you_sure:
#             logger.warning("Clear index called without confirmation. Aborting.")
#             raise ValueError("Confirmation not provided to clear index. Set are_you_sure=True.")

#         logger.warning(f"Attempting to clear index '{self.settings.INDEX_NAME}' of type '{self.settings.VECTOR_STORE_TYPE}'")
        
#         self._index = None # Dereference current index object

#         # Re-initialize LlamaSettings and our components to get a fresh state
#         # This effectively "clears" the in-memory parts and re-establishes connections
#         # For persistent stores, actual data deletion logic is needed for the store itself.

#         if self.settings.VECTOR_STORE_TYPE == "simple":
#             # For simple store, if it was persisted, delete the directory
#             # The LlamaIndex default path or a path from our StorageContext config.
#             # Assuming default "./storage" if no specific persist_dir was set in StorageContext.
#             # A more robust way would be to get persist_dir from the _storage_context if it was set.
#             persist_path = Path(self._storage_context.persist_dir if self._storage_context and self._storage_context.persist_dir else "./storage")
#             if persist_path.exists() and persist_path.is_dir():
#                 logger.info(f"Deleting SimpleVectorStore persistence directory: {persist_path}")
#                 shutil.rmtree(persist_path)
#             logger.info("SimpleVectorStore cleared (in-memory and potentially disk files).")
        
#         elif self.settings.VECTOR_STORE_TYPE == "pinecone":
#             try:
#                 # This requires pinecone client to be installed
#                 from pinecone import Pinecone as PineconeClient # Updated import
#                 pc = PineconeClient(api_key=self.settings.PINECONE_API_KEY, environment=self.settings.PINECONE_ENVIRONMENT)
#                 if self.settings.INDEX_NAME in pc.list_indexes().names:
#                     logger.info(f"Deleting Pinecone index: {self.settings.INDEX_NAME}")
#                     pc.delete_index(self.settings.INDEX_NAME)
#                     logger.info(f"Pinecone index {self.settings.INDEX_NAME} deleted.")
#                 else:
#                     logger.info(f"Pinecone index {self.settings.INDEX_NAME} not found, nothing to delete externally.")
#             except ImportError:
#                 logger.error("Pinecone client not installed. Cannot clear Pinecone index from server.")
#             except Exception as e:
#                 logger.error(f"Error clearing Pinecone index from server: {e}", exc_info=True)
#                 # Continue to re-initialize components regardless
        
#         elif self.settings.VECTOR_STORE_TYPE == "postgres":
#             # For PGVector, deleting all rows from the table is the clearest action.
#             # This needs direct DB interaction. The LlamaIndex PGVectorStore doesn't expose a "clear" method.
#             # We can attempt to get connection details from our settings.
#             logger.warning("Clearing PGVectorStore table requires direct database operation. Attempting re-initialization. Manual TRUNCATE TABLE may be needed for full clear.")
#             # A TRUNCATE would be:
#             # try:
#             #     import psycopg2 # Or asyncpg if that's what PGVectorStore uses
#             #     conn_str = f"dbname='{self.settings.PG_DB_NAME}' user='{self.settings.PG_USER}' password='{self.settings.PG_PASSWORD}' host='{self.settings.PG_HOST}' port='{self.settings.PG_PORT or 5432}'"
#             #     conn = psycopg2.connect(conn_str)
#             #     cursor = conn.cursor()
#             #     cursor.execute(f"TRUNCATE TABLE {self.settings.PG_TABLE_NAME};")
#             #     conn.commit()
#             #     logger.info(f"Successfully truncated table {self.settings.PG_TABLE_NAME} in PostgreSQL.")
#             #     cursor.close()
#             #     conn.close()
#             # except Exception as e:
#             #     logger.error(f"Failed to truncate PGVector table {self.settings.PG_TABLE_NAME}: {e}")
#             pass # For now, just re-initialize components

#         else:
#             logger.error(f"Unsupported vector store type for clearing: {self.settings.VECTOR_STORE_TYPE}")
#             # Still try to re-initialize components

#         # Re-initialize all components and LlamaSettings to ensure a fresh state
#         # This also handles re-creating index objects for Pinecone/PG if they were deleted server-side.
#         self._initialize_components_and_settings()
#         logger.info("RAGManager components and LlamaSettings re-initialized after clear attempt.")

# AGENTIC_MIRAI/app/modules/rag_manager.py
from typing import List, Optional, Any, Dict
from pathlib import Path
from llama_index.core import load_index_from_storage
import shutil

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
from llama_index.core.node_parser import NodeParser
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM as LlamaLLM # Alias to avoid conflict
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor # For type hinting
# Example Postprocessor for MMR (if you install llama-index-postprocessor-mmr or similar)
# from llama_index.postprocessor.mmr import MMRPostprocessor # Hypothetical, check actual package

# Import global LlamaIndex settings
from llama_index.core.settings import Settings as LlamaSettings

from app.factory.component_factory import ComponentFactory
from shared.config import Settings as AppSettings
from shared.log import get_logger
from shared.exceptions import DocumentProcessingError, IndexingError, QueryError, ConfigurationError, UnsupportedFeatureError
from shared.validation.query_schema import QueryRequest, QueryResponse, RetrievedNode, DocumentMetadata # QueryResponse for return type

logger = get_logger(__name__)

class RAGManager:
    def __init__(self, settings: AppSettings, component_factory: ComponentFactory):
        self.settings: AppSettings = settings
        self.factory = component_factory
        
        self._persist_directory_path: Optional[str] = None
        self._embed_model_instance: Optional[BaseEmbedding] = None
        self._llm_instance: Optional[LlamaLLM] = None
        
        self._vector_store_manager = None
        self._vector_store_llama_instance = None
        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[BaseIndex] = None

        self._initialize_components_and_settings()

    def _initialize_llm(self):
        if self.settings.LLM_PROVIDER:
            logger.info(f"Initializing LLM for LlamaIndex Settings: Provider {self.settings.LLM_PROVIDER}")
            if self.settings.LLM_PROVIDER == "openai":
                if not self.settings.OPENAI_API_KEY:
                    raise ConfigurationError("OPENAI_API_KEY must be set to use OpenAI LLM.")
                try:
                    from llama_index.llms.openai import OpenAI as OpenAI_LLM
                    self._llm_instance = OpenAI_LLM(
                        model=self.settings.OPENAI_LLM_MODEL_NAME,
                        api_key=self.settings.OPENAI_API_KEY
                    )
                    LlamaSettings.llm = self._llm_instance
                    logger.info(f"LlamaIndex global LLM set to: {self.settings.OPENAI_LLM_MODEL_NAME}")
                except ImportError:
                    logger.error("llama-index-llms-openai is not installed. Cannot initialize OpenAI LLM.")
                    raise ConfigurationError("llama-index-llms-openai is not installed.")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
                    raise ConfigurationError(f"Failed to initialize OpenAI LLM: {e}")
            else:
                logger.warning(f"LLM provider '{self.settings.LLM_PROVIDER}' not yet implemented for LlamaSettings in this RAGManager.")
        else:
            LlamaSettings.llm = None
            logger.info("No LLM provider configured; LlamaIndex global LLM set to None. Synthesis will be skipped.")

    def _initialize_components_and_settings(self):
        logger.info("Initializing RAGManager components and LlamaIndex global settings...")
        try:
            embed_model_manager = self.factory.get_embedding_model()
            self._embed_model_instance = embed_model_manager.get_embedding_model()
            LlamaSettings.embed_model = self._embed_model_instance
            logger.info(f"LlamaIndex global embedding model initialized: {type(self._embed_model_instance)}")

            self._initialize_llm()

            self._vector_store_manager = self.factory.get_vector_store()
            self._vector_store_llama_instance = self._vector_store_manager.get_vector_store()
            
            self._persist_directory_path = None
            if self.settings.VECTOR_STORE_TYPE == "simple":
                self._persist_directory_path = "./storage" 
                logger.info(f"SimpleVectorStore persistence enabled. Target directory: {self._persist_directory_path}")
                Path(self._persist_directory_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize StorageContext with persist_dir if applicable
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store_llama_instance,
                persist_dir=self._persist_directory_path # This will be None if not simple store
            )
            logger.info(f"Vector store initialized: {type(self._vector_store_llama_instance)}")
            logger.info(f"Storage context configured with vector store: {type(self._storage_context.vector_store)}"
                        f"{f' and persist_dir: {self._persist_directory_path}' if self._persist_directory_path else ''}")
            
            # Set global chunking settings (optional)
            # These are defaults; per-ingestion settings can override this via NodeParser params
            LlamaSettings.chunk_size = self.settings.DEFAULT_CHUNK_SIZE
            LlamaSettings.chunk_overlap = self.settings.DEFAULT_CHUNK_OVERLAP
            logger.info(f"LlamaIndex global chunk_size={LlamaSettings.chunk_size}, chunk_overlap={LlamaSettings.chunk_overlap}")

            self._load_or_create_index()

        except Exception as e:
            logger.error(f"Error during RAGManager component initialization: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize RAG components: {e}")

    def _load_or_create_index(self):
        logger.info(f"Attempting to load index '{self.settings.INDEX_NAME}' using vector store '{self.settings.VECTOR_STORE_TYPE}'")
        try:
            if self.settings.VECTOR_STORE_TYPE in ["pinecone", "postgres"]:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store_llama_instance,
                    storage_context=self._storage_context
                )
                logger.info(f"Connected to existing vector store '{self.settings.VECTOR_STORE_TYPE}' as index '{self.settings.INDEX_NAME}'.")
            
            elif self.settings.VECTOR_STORE_TYPE == "simple":
                can_load_from_storage = False
                if self._persist_directory_path:
                    p_dir = Path(self._persist_directory_path)
                    if (p_dir / "docstore.json").exists() and \
                       (p_dir / "index_store.json").exists() and \
                       (p_dir / "vector_store.json").exists():
                        can_load_from_storage = True
                
                if can_load_from_storage:
                    try:
                        logger.info(f"Attempting to load SimpleVectorStore index from: {self._persist_directory_path}")
                        # For loading, StorageContext needs to point to the persist_dir.
                        # LlamaSettings (embed_model, llm) will be used by load_index_from_storage.
                        loading_storage_context = StorageContext.from_defaults(persist_dir=self._persist_directory_path)
                        self._index = load_index_from_storage(loading_storage_context)
                        logger.info(f"Successfully loaded SimpleVectorStore index from {self._persist_directory_path}.")
                        if self._index and self._index.docstore:
                             logger.info(f"Loaded index has {len(self._index.docstore.docs)} documents.")
                        else:
                             logger.warning("Loaded index or its docstore is None/empty after loading from storage.")
                    except Exception as e:
                        logger.warning(f"Failed to load index from storage at {self._persist_directory_path}: {e}. Creating new index.", exc_info=True)
                        self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context)
                else:
                    logger.info(f"No existing/complete persisted data found for SimpleVectorStore at {self._persist_directory_path or 'in-memory'}. Creating new index.")
                    self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context)
            else:
                 raise ConfigurationError(f"Unsupported vector store type for index loading: {self.settings.VECTOR_STORE_TYPE}")

            if self._index is None:
                 logger.error("Index is still None after load/create attempt. This should not happen. Creating basic empty index.")
                 self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context) # Fallback with current SC
            
            logger.info(f"Index '{self.settings.INDEX_NAME}' is ready.")

        except Exception as e:
            logger.error(f"Critical error in _load_or_create_index: {e}", exc_info=True)
            self._index = VectorStoreIndex.from_documents([], storage_context=StorageContext.from_defaults()) # Absolute fallback
            raise IndexingError(f"Failed to load or create index '{self.settings.INDEX_NAME}': {e}")

    def add_document(self, file_path: str, doc_metadata: Optional[Dict[str, Any]] = None,
                     chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
        logger.info(f"Processing document: {file_path}")
        try:
            loader = self.factory.get_document_loader()
            splitter_factory = self.factory.get_text_splitter()

            # Use provided chunking params or defaults from LlamaSettings (which we set from AppSettings)
            effective_chunk_size = chunk_size if chunk_size is not None else LlamaSettings.chunk_size
            effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else LlamaSettings.chunk_overlap
            
            logger.debug(f"Using chunk_size={effective_chunk_size}, chunk_overlap={effective_chunk_overlap} for ingestion.")

            loaded_docs: List[Document] = loader.load(source=file_path, metadata=doc_metadata)
            if not loaded_docs:
                logger.warning(f"No documents loaded from {file_path}")
                return 0
            logger.info(f"Loaded {len(loaded_docs)} document parts from {file_path}.")

            node_parser: NodeParser = splitter_factory.get_node_parser(
                chunk_size=effective_chunk_size, 
                chunk_overlap=effective_chunk_overlap
            )
            nodes: List[BaseNode] = node_parser.get_nodes_from_documents(loaded_docs)
            logger.info(f"Generated {len(nodes)} nodes from document.")

            if not nodes:
                logger.warning("No nodes generated after splitting. Nothing to index.")
                return 0
            
            if self._index is None: # Should be initialized by __init__
                logger.error("Index is None during add_document. This indicates an earlier initialization failure.")
                raise IndexingError("Index not available for adding documents.")

            self._index.insert_nodes(nodes)
            logger.info(f"Successfully indexed {len(nodes)} nodes from {file_path} into '{self.settings.INDEX_NAME}'.")

            if self.settings.VECTOR_STORE_TYPE == "simple" and self._persist_directory_path:
                 logger.info(f"Persisting SimpleVectorStore to: {self._persist_directory_path}")
                 self._index.storage_context.persist(persist_dir=self._persist_directory_path)
            
            return len(nodes)
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}", exc_info=True)
            raise IndexingError(f"Failed to process and index document {file_path}: {e}")

    def query(self, request: QueryRequest) -> QueryResponse:
        logger.info(f"Received query: '{request.query_text}' with params: {request.model_dump_json(exclude_none=True)}")
        if self._index is None:
            logger.error("Index is not initialized. Cannot perform query.")
            raise QueryError("Index not initialized. Add documents first.")
        
        synthesized_answer_text: Optional[str] = None
        retrieved_nodes_output: List[RetrievedNode] = []

        try:
            retriever_manager = self.factory.get_vector_retriever()
            retriever = retriever_manager.get_retriever(self._index, request)
            logger.debug(f"Using retriever: {type(retriever)}")

            # --- Node Postprocessors (Example for MMR) ---
            node_postprocessors: List[BaseNodePostprocessor] = []
            # strategy_to_use = request.retrieval_strategy or self.settings.DEFAULT_RETRIEVER_STRATEGY
            # if strategy_to_use == "mmr":
            #     try:
            #         # You would need to install a package that provides MMRPostprocessor
            #         # e.g., `pip install llama-index-postprocessor-mmr` (hypothetical name)
            #         # from llama_index.postprocessor.mmr import MMRPostprocessor 
            #         # mmr_postprocessor = MMRPostprocessor(
            #         #     top_k=request.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K,
            #         #     # similarity_cutoff=0.5, # Optional
            #         #     # embedding_model_similarity_cutoff=0.7, # Optional for some MMR impls
            #         # )
            #         # node_postprocessors.append(mmr_postprocessor)
            #         # logger.info("MMR postprocessor added (if available).")
            #         logger.warning("MMR postprocessing requested but no MMRPostprocessor is configured/installed. Skipping.")
            #     except ImportError:
            #         logger.warning("MMRPostprocessor not found. Install relevant package for MMR postprocessing. Skipping.")
            #     except Exception as e_mmr:
            #         logger.error(f"Error initializing MMR postprocessor: {e_mmr}. Skipping.")
            # else: # For "similarity" or other strategies not needing specific postprocessors here
            #    pass

            if LlamaSettings.llm is None:
                logger.warning("No LLM configured in LlamaSettings. Performing retrieval only.")
                retrieved_nodes_output = self._retrieve_only(request) # Call helper
            else:
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=node_postprocessors if node_postprocessors else None
                    # LLM is picked from LlamaSettings.llm
                )
                logger.info(f"Querying with engine for: '{request.query_text}'")
                response_obj = query_engine.query(request.query_text)
                
                synthesized_answer_text = str(response_obj)
                if response_obj.source_nodes:
                    for node_ws in response_obj.source_nodes:
                        doc_meta = DocumentMetadata(
                            source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
                            page_number=node_ws.node.metadata.get("page_label"),
                            extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
                        )
                        retrieved_nodes_output.append(RetrievedNode(
                            text=node_ws.node.get_content(),
                            score=node_ws.score,
                            metadata=doc_meta
                        ))
                logger.info(f"Query '{request.query_text}' retrieved {len(retrieved_nodes_output)} source nodes and synthesized answer.")

            return QueryResponse(
                query_text=request.query_text,
                retrieved_nodes=retrieved_nodes_output,
                synthesized_answer=synthesized_answer_text
            )

        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            # Fallback to retrieval only if query engine fails.
            logger.warning("Falling back to retrieval-only due to query execution error.")
            try:
                retrieved_nodes_fallback = self._retrieve_only(request)
                return QueryResponse(
                    query_text=request.query_text,
                    retrieved_nodes=retrieved_nodes_fallback,
                    synthesized_answer=f"Error during synthesis/query: {str(e)}. Retrieval only."
                )
            except Exception as retrieve_e:
                logger.error(f"Error during fallback retrieval: {retrieve_e}", exc_info=True)
                raise QueryError(f"Failed to execute query '{request.query_text}' (main and fallback retrieval failed): {e}")

    def _retrieve_only(self, request: QueryRequest) -> List[RetrievedNode]:
        logger.info(f"Performing retrieval-only for query: '{request.query_text}'")
        if self._index is None: # Should not happen if checks are done before calling
            raise QueryError("Index not available for retrieval-only.")
            
        retriever_manager = self.factory.get_vector_retriever()
        retriever = retriever_manager.get_retriever(self._index, request) # Pass request for strategy/top_k
        retrieved_nodes_with_score: List[NodeWithScore] = retriever.retrieve(request.query_text)
        
        results = []
        for node_ws in retrieved_nodes_with_score:
            doc_meta = DocumentMetadata(
                source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
                page_number=node_ws.node.metadata.get("page_label"),
                extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
            )
            results.append(RetrievedNode(
                text=node_ws.node.get_content(),
                score=node_ws.score,
                metadata=doc_meta
            ))
        logger.info(f"Retrieval-only for '{request.query_text}' found {len(results)} nodes.")
        return results

    def clear_index(self, are_you_sure: bool = False):
        if not are_you_sure:
            logger.warning("Clear index called without confirmation. Aborting.")
            raise ValueError("Confirmation not provided to clear index. Set are_you_sure=True.")

        logger.warning(f"Attempting to clear index '{self.settings.INDEX_NAME}' of type '{self.settings.VECTOR_STORE_TYPE}'")
        self._index = None

        if self.settings.VECTOR_STORE_TYPE == "simple" and self._persist_directory_path:
            persist_path = Path(self._persist_directory_path)
            if persist_path.exists() and persist_path.is_dir():
                logger.info(f"Deleting SimpleVectorStore persistence directory: {persist_path}")
                shutil.rmtree(persist_path)
            logger.info("SimpleVectorStore disk persistence cleared.")
        
        elif self.settings.VECTOR_STORE_TYPE == "pinecone":
            try:
                from pinecone import Pinecone as PineconeClient
                pc = PineconeClient(api_key=self.settings.PINECONE_API_KEY, environment=self.settings.PINECONE_ENVIRONMENT)
                if self.settings.INDEX_NAME in pc.list_indexes().names: # Check if list_indexes() returns a list of IndexDescription objects
                    logger.info(f"Deleting Pinecone index: {self.settings.INDEX_NAME}")
                    pc.delete_index(self.settings.INDEX_NAME)
                    logger.info(f"Pinecone index {self.settings.INDEX_NAME} deleted.")
                else:
                    logger.info(f"Pinecone index {self.settings.INDEX_NAME} not found, nothing to delete externally.")
            except ImportError:
                logger.error("Pinecone client not installed. Cannot clear Pinecone index from server.")
            except Exception as e:
                logger.error(f"Error clearing Pinecone index from server: {e}", exc_info=True)
        
        elif self.settings.VECTOR_STORE_TYPE == "postgres":
            logger.warning("Attempting to clear PGVectorStore table. This may require direct database TRUNCATE operation for full effect if re-initialization doesn't drop/recreate.")
            # A TRUNCATE would be: (Uncomment and ensure psycopg2 is installed if you want to use this)
            # try:
            #     import psycopg2 
            #     conn_str = f"dbname='{self.settings.PG_DB_NAME}' user='{self.settings.PG_USER}' password='{self.settings.PG_PASSWORD}' host='{self.settings.PG_HOST}' port='{self.settings.PG_PORT or 5432}'"
            #     with psycopg2.connect(conn_str) as conn:
            #         with conn.cursor() as cursor:
            #             logger.info(f"Truncating table {self.settings.PG_TABLE_NAME} in PostgreSQL.")
            #             cursor.execute(f"TRUNCATE TABLE \"{self.settings.PG_TABLE_NAME}\";") # Ensure table name is quoted if needed
            #         conn.commit() # Autocommits if with block for connection used correctly
            #     logger.info(f"Successfully truncated table {self.settings.PG_TABLE_NAME} in PostgreSQL.")
            # except ImportError:
            #     logger.error("psycopg2 not installed. Cannot truncate PGVector table directly.")
            # except Exception as e:
            #     logger.error(f"Failed to truncate PGVector table {self.settings.PG_TABLE_NAME}: {e}", exc_info=True)
            pass
        
        self._initialize_components_and_settings() # Re-initialize to get a fresh state
        logger.info("RAGManager components and LlamaSettings re-initialized after clear attempt.")