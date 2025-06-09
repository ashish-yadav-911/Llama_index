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

# Import global LlamaIndex settings
from llama_index.core.settings import Settings as LlamaSettings

from app.factory.component_factory import ComponentFactory
from shared.config import Settings as AppSettings
from shared.log import get_logger
from shared.exceptions import DocumentProcessingError, IndexingError, QueryError, ConfigurationError, UnsupportedFeatureError
from shared.validation.query_schema import QueryRequest, QueryResponse, RetrievedNode, DocumentMetadata

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
            """Initializes the LLM based on configuration and sets it in LlamaSettings."""
            if self.settings.LLM_PROVIDER:
                logger.info(f"Initializing LLM for LlamaIndex Settings: Provider {self.settings.LLM_PROVIDER}")
                if self.settings.LLM_PROVIDER == "openai":
                    if not self.settings.OPENAI_API_KEY:
                        raise ConfigurationError("OPENAI_API_KEY must be set to use OpenAI LLM.")
                    try:
                        from llama_index.llms.openai import OpenAI as OpenAI_LLM
                        
                        # Provide defaults for fields that might be causing issues
                        # For 'logprobs', False is a common default if you don't need them.
                        # For 'default_headers', an empty dict is a safe default.
                        self._llm_instance = OpenAI_LLM(
                            model=self.settings.OPENAI_LLM_MODEL_NAME,
                            api_key=self.settings.OPENAI_API_KEY,
                            temperature=self.settings.OPENAI_LLM_TEMPERATURE, # Pass temperature from your config
                            # --- ADDED/MODIFIED DEFAULTS ---
                            logprobs=False,  # Explicitly set logprobs if it's required
                            default_headers={} # Provide an empty dict for default_headers
                            # You might also need to set other fields if more errors appear,
                            # e.g., max_tokens, etc., though LlamaIndex usually has sensible defaults.
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
            
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store_llama_instance,
                persist_dir=self._persist_directory_path
            )
            logger.info(f"Vector store initialized: {type(self._vector_store_llama_instance)}")
            logger.info(f"Storage context configured with vector store: {type(self._storage_context.vector_store)}"
                        f"{f' and persist_dir: {self._persist_directory_path}' if self._persist_directory_path else ''}")
            
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
                 self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context)
            
            logger.info(f"Index '{self.settings.INDEX_NAME}' is ready.")

        except Exception as e:
            logger.error(f"Critical error in _load_or_create_index: {e}", exc_info=True)
            self._index = VectorStoreIndex.from_documents([], storage_context=StorageContext.from_defaults())
            raise IndexingError(f"Failed to load or create index '{self.settings.INDEX_NAME}': {e}")

    def add_document(self, file_path: str, doc_metadata: Optional[Dict[str, Any]] = None,
                     chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
        logger.info(f"Processing document: {file_path}")
        try:
            loader = self.factory.get_document_loader()
            splitter_factory = self.factory.get_text_splitter()

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
            
            if self._index is None:
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

    def _format_retrieved_nodes(self, nodes_with_score: List[NodeWithScore]) -> List[RetrievedNode]:
        output: List[RetrievedNode] = []
        for node_ws in nodes_with_score:
            doc_meta = DocumentMetadata(
                source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
                page_number=node_ws.node.metadata.get("page_label"),
                extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
            )
            output.append(RetrievedNode(
                text=node_ws.node.get_content(),
                score=node_ws.score,
                metadata=doc_meta
            ))
        return output

    def _apply_postprocessors(self, nodes: List[NodeWithScore], postprocessors: List[BaseNodePostprocessor], query_bundle: QueryBundle) -> List[NodeWithScore]:
        processed_nodes = nodes
        for pp in postprocessors:
            logger.info(f"Applying postprocessor: {type(pp).__name__}")
            processed_nodes = pp.postprocess_nodes(processed_nodes, query_bundle=query_bundle)
        return processed_nodes

    def _retrieve_only_simple_fallback(self, request: QueryRequest) -> List[RetrievedNode]:
        logger.info(f"Performing SIMPLIFIED retrieval-only fallback for query: '{request.query_text}'")
        if self._index is None:
            logger.error("Index is None during simple_fallback retrieval.")
            raise QueryError("Index not available for simple fallback retrieval.")
        
        basic_retriever = self._index.as_retriever(
            similarity_top_k=(request.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K)
        )
        retrieved_nodes_with_score: List[NodeWithScore] = basic_retriever.retrieve(request.query_text)
        return self._format_retrieved_nodes(retrieved_nodes_with_score)

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
            logger.debug(f"Using retriever: {type(retriever).__name__} for strategy '{request.retrieval_strategy or self.settings.DEFAULT_RETRIEVER_STRATEGY}'")

            # Node Postprocessors list - can be populated if specific postprocessing is needed (e.g., a custom reranker)
            node_postprocessors: List[BaseNodePostprocessor] = []
            # If you add a reranker or other general postprocessor, add it to 'node_postprocessors' here.
            # For example:
            # if self.settings.ENABLE_RERANKER: # Assuming a setting
            #     try:
            #         from llama_index.core.postprocessor import SentenceTransformerRerank 
            #         reranker = SentenceTransformerRerank(model="...", top_n=...)
            #         node_postprocessors.append(reranker)
            #         logger.info("Reranker postprocessor added.")
            #     except ImportError: logger.warning("Reranker desired but sentence-transformers not installed.")

            if LlamaSettings.llm is None:
                logger.warning("No LLM configured in LlamaSettings. Performing retrieval only.")
                retrieved_nodes_with_score: List[NodeWithScore] = retriever.retrieve(request.query_text)
                if node_postprocessors:
                    query_bundle = QueryBundle(request.query_text)
                    retrieved_nodes_with_score = self._apply_postprocessors(retrieved_nodes_with_score, node_postprocessors, query_bundle)
                retrieved_nodes_output = self._format_retrieved_nodes(retrieved_nodes_with_score)
                synthesized_answer_text = "Retrieval complete. LLM not configured for synthesis."
                logger.info(f"Retrieval-only for '{request.query_text}' yielded {len(retrieved_nodes_output)} nodes.")
            else:
                logger.info(f"LLM '{LlamaSettings.llm.metadata.model_name}' configured. Proceeding with synthesis.")
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=node_postprocessors if node_postprocessors else None
                )
                logger.info(f"Querying with engine for: '{request.query_text}'")
                response_obj = query_engine.query(request.query_text)
                
                synthesized_answer_text = str(response_obj)
                if response_obj.source_nodes:
                    retrieved_nodes_output = self._format_retrieved_nodes(response_obj.source_nodes)
                logger.info(f"Query '{request.query_text}' - Retrieved {len(retrieved_nodes_output)} source nodes. Synthesized answer: '{synthesized_answer_text[:100]}...'")

            return QueryResponse(
                query_text=request.query_text,
                retrieved_nodes=retrieved_nodes_output,
                synthesized_answer=synthesized_answer_text
            )
        except UnsupportedFeatureError as ufe:
            logger.error(f"Unsupported feature used: {ufe}", exc_info=True)
            raise QueryError(f"Query failed due to unsupported feature: {ufe.message}")
        except ConfigurationError as ce:
            logger.error(f"Configuration error during query: {ce}", exc_info=True)
            raise QueryError(f"Query failed due to configuration error: {ce.message}")
        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            logger.warning("Falling back to simplified retrieval-only due to query execution error.")
            try:
                retrieved_nodes_fallback = self._retrieve_only_simple_fallback(request)
                return QueryResponse(
                    query_text=request.query_text,
                    retrieved_nodes=retrieved_nodes_fallback,
                    synthesized_answer=f"Error during main query processing: {str(e)}. Retrieval only."
                )
            except Exception as retrieve_e:
                logger.error(f"Error during fallback retrieval: {retrieve_e}", exc_info=True)
                raise QueryError(f"Failed to execute query '{request.query_text}'. Main query error: '{str(e)}'. Fallback retrieval error: '{str(retrieve_e)}'.")

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
                # Check if list_indexes() returns a list of IndexDescription objects which have a 'name' attribute
                # or if it's a list of strings directly (depends on pinecone-client version)
                index_names = []
                index_list_response = pc.list_indexes()
                if hasattr(index_list_response, 'names'): # For newer client versions returning an object with a names list
                    index_names = index_list_response.names
                elif isinstance(index_list_response, list) and all(isinstance(i, str) for i in index_list_response): # Older list of strings
                    index_names = index_list_response
                elif isinstance(index_list_response, list) and index_list_response and hasattr(index_list_response[0], 'name'): # List of IndexDescription like objects
                    index_names = [idx_desc.name for idx_desc in index_list_response]


                if self.settings.INDEX_NAME in index_names:
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
            logger.warning("Clearing PGVectorStore table. For a full clear, manual TRUNCATE TABLE in the DB might be needed if re-initialization doesn't handle it.")
            # If you want to enable direct TRUNCATE:
            # 1. pip install psycopg2-binary
            # 2. Uncomment the following block:
            # try:
            #     import psycopg2 
            #     conn_str = f"dbname='{self.settings.PG_DB_NAME}' user='{self.settings.PG_USER}' password='{self.settings.PG_PASSWORD}' host='{self.settings.PG_HOST}' port='{self.settings.PG_PORT or 5432}'"
            #     with psycopg2.connect(conn_str) as conn: # 'with' handles conn.close()
            #         with conn.cursor() as cursor: # 'with' handles cursor.close()
            #             logger.info(f"Truncating table \"{self.settings.PG_TABLE_NAME}\" in PostgreSQL.")
            #             # Ensure table name is quoted if it contains special characters or capitals
            #             cursor.execute(f"TRUNCATE TABLE \"{self.settings.PG_TABLE_NAME}\";") 
            #         conn.commit() 
            #     logger.info(f"Successfully truncated table \"{self.settings.PG_TABLE_NAME}\" in PostgreSQL.")
            # except ImportError:
            #     logger.error("psycopg2-binary not installed. Cannot truncate PGVector table directly.")
            # except Exception as e:
            #     logger.error(f"Failed to truncate PGVector table \"{self.settings.PG_TABLE_NAME}\": {e}", exc_info=True)
            pass # Current behavior: rely on re-initialization.
        
        self._initialize_components_and_settings()
        logger.info("RAGManager components and LlamaSettings re-initialized after clear attempt.")




        


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
# from llama_index.core.postprocessor.types import BaseNodePostprocessor # For type hinting
# # Example Postprocessor for MMR (if you install llama-index-postprocessor-mmr or similar)
# # from llama_index.postprocessor.mmr import MMRPostprocessor # Hypothetical, check actual package

# # Import global LlamaIndex settings
# from llama_index.core.settings import Settings as LlamaSettings

# from app.factory.component_factory import ComponentFactory
# from shared.config import Settings as AppSettings
# from shared.log import get_logger
# from shared.exceptions import DocumentProcessingError, IndexingError, QueryError, ConfigurationError, UnsupportedFeatureError
# from shared.validation.query_schema import QueryRequest, QueryResponse, RetrievedNode, DocumentMetadata # QueryResponse for return type

# logger = get_logger(__name__)

# class RAGManager:
#     def __init__(self, settings: AppSettings, component_factory: ComponentFactory):
#         self.settings: AppSettings = settings
#         self.factory = component_factory
        
#         self._persist_directory_path: Optional[str] = None
#         self._embed_model_instance: Optional[BaseEmbedding] = None
#         self._llm_instance: Optional[LlamaLLM] = None
        
#         self._vector_store_manager = None
#         self._vector_store_llama_instance = None
#         self._storage_context: Optional[StorageContext] = None
#         self._index: Optional[BaseIndex] = None

#         self._initialize_components_and_settings()

#     def _initialize_llm(self):
#         if self.settings.LLM_PROVIDER:
#             logger.info(f"Initializing LLM for LlamaIndex Settings: Provider {self.settings.LLM_PROVIDER}")
#             if self.settings.LLM_PROVIDER == "openai":
#                 if not self.settings.OPENAI_API_KEY:
#                     raise ConfigurationError("OPENAI_API_KEY must be set to use OpenAI LLM.")
#                 try:
#                     from llama_index.llms.openai import OpenAI as OpenAI_LLM
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
#             LlamaSettings.llm = None
#             logger.info("No LLM provider configured; LlamaIndex global LLM set to None. Synthesis will be skipped.")

#     def _initialize_components_and_settings(self):
#         logger.info("Initializing RAGManager components and LlamaIndex global settings...")
#         try:
#             embed_model_manager = self.factory.get_embedding_model()
#             self._embed_model_instance = embed_model_manager.get_embedding_model()
#             LlamaSettings.embed_model = self._embed_model_instance
#             logger.info(f"LlamaIndex global embedding model initialized: {type(self._embed_model_instance)}")

#             self._initialize_llm()

#             self._vector_store_manager = self.factory.get_vector_store()
#             self._vector_store_llama_instance = self._vector_store_manager.get_vector_store()
            
#             self._persist_directory_path = None
#             if self.settings.VECTOR_STORE_TYPE == "simple":
#                 self._persist_directory_path = "./storage" 
#                 logger.info(f"SimpleVectorStore persistence enabled. Target directory: {self._persist_directory_path}")
#                 Path(self._persist_directory_path).mkdir(parents=True, exist_ok=True)
            
#             # Initialize StorageContext with persist_dir if applicable
#             self._storage_context = StorageContext.from_defaults(
#                 vector_store=self._vector_store_llama_instance,
#                 persist_dir=self._persist_directory_path # This will be None if not simple store
#             )
#             logger.info(f"Vector store initialized: {type(self._vector_store_llama_instance)}")
#             logger.info(f"Storage context configured with vector store: {type(self._storage_context.vector_store)}"
#                         f"{f' and persist_dir: {self._persist_directory_path}' if self._persist_directory_path else ''}")
            
#             # Set global chunking settings (optional)
#             # These are defaults; per-ingestion settings can override this via NodeParser params
#             LlamaSettings.chunk_size = self.settings.DEFAULT_CHUNK_SIZE
#             LlamaSettings.chunk_overlap = self.settings.DEFAULT_CHUNK_OVERLAP
#             logger.info(f"LlamaIndex global chunk_size={LlamaSettings.chunk_size}, chunk_overlap={LlamaSettings.chunk_overlap}")

#             self._load_or_create_index()

#         except Exception as e:
#             logger.error(f"Error during RAGManager component initialization: {e}", exc_info=True)
#             raise ConfigurationError(f"Failed to initialize RAG components: {e}")

#     def _load_or_create_index(self):
#         logger.info(f"Attempting to load index '{self.settings.INDEX_NAME}' using vector store '{self.settings.VECTOR_STORE_TYPE}'")
#         try:
#             if self.settings.VECTOR_STORE_TYPE in ["pinecone", "postgres"]:
#                 self._index = VectorStoreIndex.from_vector_store(
#                     vector_store=self._vector_store_llama_instance,
#                     storage_context=self._storage_context
#                 )
#                 logger.info(f"Connected to existing vector store '{self.settings.VECTOR_STORE_TYPE}' as index '{self.settings.INDEX_NAME}'.")
            
#             elif self.settings.VECTOR_STORE_TYPE == "simple":
#                 can_load_from_storage = False
#                 if self._persist_directory_path:
#                     p_dir = Path(self._persist_directory_path)
#                     if (p_dir / "docstore.json").exists() and \
#                        (p_dir / "index_store.json").exists() and \
#                        (p_dir / "vector_store.json").exists():
#                         can_load_from_storage = True
                
#                 if can_load_from_storage:
#                     try:
#                         logger.info(f"Attempting to load SimpleVectorStore index from: {self._persist_directory_path}")
#                         # For loading, StorageContext needs to point to the persist_dir.
#                         # LlamaSettings (embed_model, llm) will be used by load_index_from_storage.
#                         loading_storage_context = StorageContext.from_defaults(persist_dir=self._persist_directory_path)
#                         self._index = load_index_from_storage(loading_storage_context)
#                         logger.info(f"Successfully loaded SimpleVectorStore index from {self._persist_directory_path}.")
#                         if self._index and self._index.docstore:
#                              logger.info(f"Loaded index has {len(self._index.docstore.docs)} documents.")
#                         else:
#                              logger.warning("Loaded index or its docstore is None/empty after loading from storage.")
#                     except Exception as e:
#                         logger.warning(f"Failed to load index from storage at {self._persist_directory_path}: {e}. Creating new index.", exc_info=True)
#                         self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context)
#                 else:
#                     logger.info(f"No existing/complete persisted data found for SimpleVectorStore at {self._persist_directory_path or 'in-memory'}. Creating new index.")
#                     self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context)
#             else:
#                  raise ConfigurationError(f"Unsupported vector store type for index loading: {self.settings.VECTOR_STORE_TYPE}")

#             if self._index is None:
#                  logger.error("Index is still None after load/create attempt. This should not happen. Creating basic empty index.")
#                  self._index = VectorStoreIndex.from_documents([], storage_context=self._storage_context) # Fallback with current SC
            
#             logger.info(f"Index '{self.settings.INDEX_NAME}' is ready.")

#         except Exception as e:
#             logger.error(f"Critical error in _load_or_create_index: {e}", exc_info=True)
#             self._index = VectorStoreIndex.from_documents([], storage_context=StorageContext.from_defaults()) # Absolute fallback
#             raise IndexingError(f"Failed to load or create index '{self.settings.INDEX_NAME}': {e}")

#     def add_document(self, file_path: str, doc_metadata: Optional[Dict[str, Any]] = None,
#                      chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
#         logger.info(f"Processing document: {file_path}")
#         try:
#             loader = self.factory.get_document_loader()
#             splitter_factory = self.factory.get_text_splitter()

#             # Use provided chunking params or defaults from LlamaSettings (which we set from AppSettings)
#             effective_chunk_size = chunk_size if chunk_size is not None else LlamaSettings.chunk_size
#             effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else LlamaSettings.chunk_overlap
            
#             logger.debug(f"Using chunk_size={effective_chunk_size}, chunk_overlap={effective_chunk_overlap} for ingestion.")

#             loaded_docs: List[Document] = loader.load(source=file_path, metadata=doc_metadata)
#             if not loaded_docs:
#                 logger.warning(f"No documents loaded from {file_path}")
#                 return 0
#             logger.info(f"Loaded {len(loaded_docs)} document parts from {file_path}.")

#             node_parser: NodeParser = splitter_factory.get_node_parser(
#                 chunk_size=effective_chunk_size, 
#                 chunk_overlap=effective_chunk_overlap
#             )
#             nodes: List[BaseNode] = node_parser.get_nodes_from_documents(loaded_docs)
#             logger.info(f"Generated {len(nodes)} nodes from document.")

#             if not nodes:
#                 logger.warning("No nodes generated after splitting. Nothing to index.")
#                 return 0
            
#             if self._index is None: # Should be initialized by __init__
#                 logger.error("Index is None during add_document. This indicates an earlier initialization failure.")
#                 raise IndexingError("Index not available for adding documents.")

#             self._index.insert_nodes(nodes)
#             logger.info(f"Successfully indexed {len(nodes)} nodes from {file_path} into '{self.settings.INDEX_NAME}'.")

#             if self.settings.VECTOR_STORE_TYPE == "simple" and self._persist_directory_path:
#                  logger.info(f"Persisting SimpleVectorStore to: {self._persist_directory_path}")
#                  self._index.storage_context.persist(persist_dir=self._persist_directory_path)
            
#             return len(nodes)
#         except Exception as e:
#             logger.error(f"Error adding document {file_path}: {e}", exc_info=True)
#             raise IndexingError(f"Failed to process and index document {file_path}: {e}")

#     def query(self, request: QueryRequest) -> QueryResponse:
#         logger.info(f"Received query: '{request.query_text}' with params: {request.model_dump_json(exclude_none=True)}")
#         if self._index is None:
#             logger.error("Index is not initialized. Cannot perform query.")
#             raise QueryError("Index not initialized. Add documents first.")
        
#         synthesized_answer_text: Optional[str] = None
#         retrieved_nodes_output: List[RetrievedNode] = []

#         try:
#             retriever_manager = self.factory.get_vector_retriever()
#             retriever = retriever_manager.get_retriever(self._index, request)
#             logger.debug(f"Using retriever: {type(retriever)}")

#             # --- Node Postprocessors (Example for MMR) ---
#             node_postprocessors: List[BaseNodePostprocessor] = []
#             # strategy_to_use = request.retrieval_strategy or self.settings.DEFAULT_RETRIEVER_STRATEGY
#             # if strategy_to_use == "mmr":
#             #     try:
#             #         # You would need to install a package that provides MMRPostprocessor
#             #         # e.g., `pip install llama-index-postprocessor-mmr` (hypothetical name)
#             #         # from llama_index.postprocessor.mmr import MMRPostprocessor 
#             #         # mmr_postprocessor = MMRPostprocessor(
#             #         #     top_k=request.top_k or self.settings.DEFAULT_RETRIEVER_TOP_K,
#             #         #     # similarity_cutoff=0.5, # Optional
#             #         #     # embedding_model_similarity_cutoff=0.7, # Optional for some MMR impls
#             #         # )
#             #         # node_postprocessors.append(mmr_postprocessor)
#             #         # logger.info("MMR postprocessor added (if available).")
#             #         logger.warning("MMR postprocessing requested but no MMRPostprocessor is configured/installed. Skipping.")
#             #     except ImportError:
#             #         logger.warning("MMRPostprocessor not found. Install relevant package for MMR postprocessing. Skipping.")
#             #     except Exception as e_mmr:
#             #         logger.error(f"Error initializing MMR postprocessor: {e_mmr}. Skipping.")
#             # else: # For "similarity" or other strategies not needing specific postprocessors here
#             #    pass

#             if LlamaSettings.llm is None:
#                 logger.warning("No LLM configured in LlamaSettings. Performing retrieval only.")
#                 retrieved_nodes_output = self._retrieve_only(request) # Call helper
#             else:
#                 query_engine = RetrieverQueryEngine.from_args(
#                     retriever=retriever,
#                     node_postprocessors=node_postprocessors if node_postprocessors else None
#                     # LLM is picked from LlamaSettings.llm
#                 )
#                 logger.info(f"Querying with engine for: '{request.query_text}'")
#                 response_obj = query_engine.query(request.query_text)
                
#                 synthesized_answer_text = str(response_obj)
#                 if response_obj.source_nodes:
#                     for node_ws in response_obj.source_nodes:
#                         doc_meta = DocumentMetadata(
#                             source=node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source"),
#                             page_number=node_ws.node.metadata.get("page_label"),
#                             extra_info={k: v for k, v in node_ws.node.metadata.items() if k not in ["file_name", "page_label", "source"]}
#                         )
#                         retrieved_nodes_output.append(RetrievedNode(
#                             text=node_ws.node.get_content(),
#                             score=node_ws.score,
#                             metadata=doc_meta
#                         ))
#                 logger.info(f"Query '{request.query_text}' retrieved {len(retrieved_nodes_output)} source nodes and synthesized answer.")

#             return QueryResponse(
#                 query_text=request.query_text,
#                 retrieved_nodes=retrieved_nodes_output,
#                 synthesized_answer=synthesized_answer_text
#             )

#         except Exception as e:
#             logger.error(f"Error during query execution: {e}", exc_info=True)
#             # Fallback to retrieval only if query engine fails.
#             logger.warning("Falling back to retrieval-only due to query execution error.")
#             try:
#                 retrieved_nodes_fallback = self._retrieve_only(request)
#                 return QueryResponse(
#                     query_text=request.query_text,
#                     retrieved_nodes=retrieved_nodes_fallback,
#                     synthesized_answer=f"Error during synthesis/query: {str(e)}. Retrieval only."
#                 )
#             except Exception as retrieve_e:
#                 logger.error(f"Error during fallback retrieval: {retrieve_e}", exc_info=True)
#                 raise QueryError(f"Failed to execute query '{request.query_text}' (main and fallback retrieval failed): {e}")

#     def _retrieve_only(self, request: QueryRequest) -> List[RetrievedNode]:
#         logger.info(f"Performing retrieval-only for query: '{request.query_text}'")
#         if self._index is None: # Should not happen if checks are done before calling
#             raise QueryError("Index not available for retrieval-only.")
            
#         retriever_manager = self.factory.get_vector_retriever()
#         retriever = retriever_manager.get_retriever(self._index, request) # Pass request for strategy/top_k
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
#         self._index = None

#         if self.settings.VECTOR_STORE_TYPE == "simple" and self._persist_directory_path:
#             persist_path = Path(self._persist_directory_path)
#             if persist_path.exists() and persist_path.is_dir():
#                 logger.info(f"Deleting SimpleVectorStore persistence directory: {persist_path}")
#                 shutil.rmtree(persist_path)
#             logger.info("SimpleVectorStore disk persistence cleared.")
        
#         elif self.settings.VECTOR_STORE_TYPE == "pinecone":
#             try:
#                 from pinecone import Pinecone as PineconeClient
#                 pc = PineconeClient(api_key=self.settings.PINECONE_API_KEY, environment=self.settings.PINECONE_ENVIRONMENT)
#                 if self.settings.INDEX_NAME in pc.list_indexes().names: # Check if list_indexes() returns a list of IndexDescription objects
#                     logger.info(f"Deleting Pinecone index: {self.settings.INDEX_NAME}")
#                     pc.delete_index(self.settings.INDEX_NAME)
#                     logger.info(f"Pinecone index {self.settings.INDEX_NAME} deleted.")
#                 else:
#                     logger.info(f"Pinecone index {self.settings.INDEX_NAME} not found, nothing to delete externally.")
#             except ImportError:
#                 logger.error("Pinecone client not installed. Cannot clear Pinecone index from server.")
#             except Exception as e:
#                 logger.error(f"Error clearing Pinecone index from server: {e}", exc_info=True)
        
#         elif self.settings.VECTOR_STORE_TYPE == "postgres":
#             logger.warning("Attempting to clear PGVectorStore table. This may require direct database TRUNCATE operation for full effect if re-initialization doesn't drop/recreate.")
#             # A TRUNCATE would be: (Uncomment and ensure psycopg2 is installed if you want to use this)
#             # try:
#             #     import psycopg2 
#             #     conn_str = f"dbname='{self.settings.PG_DB_NAME}' user='{self.settings.PG_USER}' password='{self.settings.PG_PASSWORD}' host='{self.settings.PG_HOST}' port='{self.settings.PG_PORT or 5432}'"
#             #     with psycopg2.connect(conn_str) as conn:
#             #         with conn.cursor() as cursor:
#             #             logger.info(f"Truncating table {self.settings.PG_TABLE_NAME} in PostgreSQL.")
#             #             cursor.execute(f"TRUNCATE TABLE \"{self.settings.PG_TABLE_NAME}\";") # Ensure table name is quoted if needed
#             #         conn.commit() # Autocommits if with block for connection used correctly
#             #     logger.info(f"Successfully truncated table {self.settings.PG_TABLE_NAME} in PostgreSQL.")
#             # except ImportError:
#             #     logger.error("psycopg2 not installed. Cannot truncate PGVector table directly.")
#             # except Exception as e:
#             #     logger.error(f"Failed to truncate PGVector table {self.settings.PG_TABLE_NAME}: {e}", exc_info=True)
#             pass
        
#         self._initialize_components_and_settings() # Re-initialize to get a fresh state
#         logger.info("RAGManager components and LlamaSettings re-initialized after clear attempt.")