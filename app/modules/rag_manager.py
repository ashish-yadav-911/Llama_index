# AGENTIC_MIRAI/app/modules/rag_manager.py
from typing import List, Optional, Any, Dict
from pathlib import Path
import shutil

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.indices.base import BaseIndex
from llama_index.core import Settings as LlamaSettings # Renamed to avoid conflict with your Settings
from llama_index.core.node_parser import NodeParser # For type hinting

from app.factory.component_factory import ComponentFactory
from shared.config import Settings as AppSettings # Renamed to avoid conflict
from shared.log import get_logger
from shared.exceptions import DocumentProcessingError, IndexingError, QueryError, ConfigurationError
from shared.validation.query_schema import QueryRequest, RetrievedNode, DocumentMetadata

logger = get_logger(__name__)

class RAGManager:
    def __init__(self, settings: AppSettings, component_factory: ComponentFactory): # Use AppSettings
        self.settings = settings # This is your application's config
        self.factory = component_factory
        
        # No self._service_context anymore
        self._embed_model_instance = None
        self._vector_store_instance_manager = None
        self._vector_store_llama_instance = None
        self._storage_context = None
        self._index: Optional[BaseIndex] = None # Type hint for clarity

        self._initialize_components()

    def _initialize_components(self):
        logger.info("Initializing RAGManager components using new LlamaIndex Settings API...")
        try:
            # Embedding Model
            embed_model_manager = self.factory.get_embedding_model()
            self._embed_model_instance = embed_model_manager.get_embedding_model()
            LlamaSettings.embed_model = self._embed_model_instance # Set globally for LlamaIndex
            logger.info(f"LlamaIndex global embed_model initialized: {type(LlamaSettings.embed_model)}")

            # LLM (Optional, for future synthesis or some node parsers)
            # If you had an LLM:
            # llm_manager = self.factory.get_llm() # You'd create this in factory
            # self._llm_instance = llm_manager.get_llm()
            # LlamaSettings.llm = self._llm_instance
            # logger.info(f"LlamaIndex global llm initialized: {type(LlamaSettings.llm)}")
            LlamaSettings.llm = None # Explicitly None if not used for now

            # Node Parser (can be set globally or per operation)
            # For default chunking from settings, we can set a default node parser globally
            # Or create it per ingestion if chunk_size/overlap varies
            # Let's set a default one here, can be overridden in add_document
            text_splitter = self.factory.get_text_splitter()
            default_node_parser = text_splitter.get_node_parser(
                chunk_size=self.settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=self.settings.DEFAULT_CHUNK_OVERLAP
            )
            LlamaSettings.node_parser = default_node_parser
            logger.info(f"LlamaIndex global node_parser initialized with default chunk settings: {type(LlamaSettings.node_parser)}")


            # Vector Store
            self._vector_store_instance_manager = self.factory.get_vector_store()
            self._vector_store_llama_instance = self._vector_store_instance_manager.get_vector_store()
            self._storage_context = self._vector_store_instance_manager.get_storage_context()
            logger.info(f"Vector store initialized: {type(self._vector_store_llama_instance)}")
            logger.info(f"Storage context configured with: {type(self._storage_context.vector_store)}")

            self._load_or_create_index()

        except Exception as e:
            logger.error(f"Error during RAGManager component initialization: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize RAG components: {e}")


    def _load_or_create_index(self):
        logger.info(f"Attempting to load index '{self.settings.INDEX_NAME}' from vector store '{self.settings.VECTOR_STORE_TYPE}'")
        try:
            # With the new Settings API, components like embed_model are picked from LlamaSettings
            # or can be passed directly if the constructor supports it.
            # VectorStoreIndex.from_vector_store uses the global LlamaSettings.embed_model
            
            # Check if index exists (this logic is store-dependent and can be tricky)
            # For now, assume from_vector_store handles connecting or creating an interface to it.
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store_llama_instance,
                storage_context=self._storage_context # Still useful for some stores
                # service_context is no longer used here
            )
            logger.info(f"Successfully connected to vector store '{self.settings.VECTOR_STORE_TYPE}' as index '{self.settings.INDEX_NAME}'.")
            # If from_vector_store doesn't create an "empty" index object for an empty remote store,
            # and returns None or raises an error, you might need to catch that and create an empty index.
            # However, typically it returns an index object ready to be populated.

            # If it's a truly new/empty simple vector store, we might still want to initialize it differently.
            # The current from_vector_store for SimpleVectorStore should return a usable index.
            # If not, an explicit VectorStoreIndex([], storage_context=self._storage_context) might be needed.

        except Exception as e: # Broad exception for now
            logger.warning(f"Could not load index directly from vector store (maybe it's empty or first time): {e}. Attempting to create new index object.", exc_info=True)
            # If loading fails (e.g., store is truly empty and from_vector_store doesn't handle it gracefully by returning an empty Index)
            # Create a new index object. It will use the global LlamaSettings.
            try:
                self._index = VectorStoreIndex(
                    nodes=[], # Start with no nodes
                    storage_context=self._storage_context
                )
                logger.info(f"Created a new, empty index object for '{self.settings.INDEX_NAME}'.")
            except Exception as e_create:
                 logger.error(f"Failed to create a new empty index: {e_create}", exc_info=True)
                 raise IndexingError(f"Failed to load or create index '{self.settings.INDEX_NAME}': {e_create}")
            
        if self._index is None: # Should not happen if the above logic is correct
            logger.error("Index is still None after load/create attempts. This should not happen.")
            raise IndexingError(f"Failed to initialize index object for '{self.settings.INDEX_NAME}'.")
            
        logger.info(f"Index '{self.settings.INDEX_NAME}' is ready.")


    def add_document(self, file_path: str, doc_metadata: Optional[Dict[str, Any]] = None,
                     chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
        logger.info(f"Processing document: {file_path}")
        if self._index is None:
            logger.error("Index is not available for add_document.")
            raise IndexingError("Cannot add document: Index is not initialized.")
            
        try:
            loader = self.factory.get_document_loader()
            splitter = self.factory.get_text_splitter()

            effective_chunk_size = chunk_size if chunk_size is not None else self.settings.DEFAULT_CHUNK_SIZE
            effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.DEFAULT_CHUNK_OVERLAP
            
            logger.debug(f"Using chunk_size={effective_chunk_size}, chunk_overlap={effective_chunk_overlap}")

            loaded_docs = loader.load(source=file_path, metadata=doc_metadata)
            if not loaded_docs:
                logger.warning(f"No documents loaded from {file_path}")
                return 0
            logger.info(f"Loaded {len(loaded_docs)} document parts from {file_path}.")

            # Node Parser for this specific ingestion
            # If chunk params are different from global, create a specific node_parser
            current_node_parser: NodeParser
            if chunk_size is not None or chunk_overlap is not None:
                current_node_parser = splitter.get_node_parser(
                    chunk_size=effective_chunk_size, 
                    chunk_overlap=effective_chunk_overlap
                )
                logger.info("Using custom node_parser for this ingestion.")
            else:
                current_node_parser = LlamaSettings.node_parser # Use the global default
                logger.info("Using global LlamaSettings.node_parser for this ingestion.")

            # Generate nodes using the selected node parser
            nodes = current_node_parser.get_nodes_from_documents(loaded_docs, show_progress=True)
            logger.info(f"Generated {len(nodes)} nodes from document.")

            if not nodes:
                logger.warning("No nodes generated after splitting. Nothing to index.")
                return 0

            # Insert nodes. This will use LlamaSettings.embed_model for embeddings
            # and the index's configured vector_store (via storage_context).
            self._index.insert_nodes(nodes, show_progress=True)
            
            logger.info(f"Successfully indexed {len(nodes)} nodes from {file_path} into '{self.settings.INDEX_NAME}'.")

            # Persistence for SimpleVectorStore if needed (often manual or via StorageContext)
            if self.settings.VECTOR_STORE_TYPE == "simple":
                if hasattr(self._storage_context, 'persist'):
                    # Default persist_dir is ./storage if not specified in StorageContext
                    persist_dir = getattr(self._storage_context, '_persist_dir', './storage')
                    self._storage_context.persist(persist_dir=persist_dir)
                    logger.info(f"Persisted SimpleVectorStore to {persist_dir}.")
            return len(nodes)

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}", exc_info=True)
            raise IndexingError(f"Failed to process and index document {file_path}: {e}")


    def query(self, request: QueryRequest) -> List[RetrievedNode]:
        logger.info(f"Received query: '{request.query_text}' with params: {request.model_dump_json(exclude_none=True)}")
        if self._index is None:
            logger.error("Index is not initialized. Cannot perform query.")
            raise QueryError("Index not initialized. Add documents first.")

        try:
            retriever_manager = self.factory.get_vector_retriever()
            # The retriever is configured based on QueryRequest parameters AND the index itself
            # get_retriever might need to be adjusted if it used service_context
            retriever = retriever_manager.get_retriever(self._index, request) 
            # as_retriever() will use the global LlamaSettings (embed_model for query embedding)
            
            logger.debug(f"Using retriever: {type(retriever)}")
            
            # Postprocessors can be added to the retriever or query engine if needed
            # For MMR, the LlamaIndexVectorRetriever might need adjustment or a postprocessor.
            # If `retriever_manager.get_retriever` returns a retriever from `index.as_retriever`,
            # then `similarity_top_k`, etc. are handled there.
            # For MMR, `index.as_retriever(vector_store_query_mode="mmr", ...)` might be an option,
            # or use a NodePostprocessor.

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
            
            logger.info(f"Query '{request.query_text}' retrieved {len(results)} nodes.")
            return results

        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            raise QueryError(f"Failed to execute query '{request.query_text}': {e}")

    def clear_index(self, are_you_sure: bool = False):
        # ... (This method might not need many changes unless it directly used service_context)
        # The main thing is re-initializing components, which now uses LlamaSettings.
        if not are_you_sure:
            logger.warning("Clear index called without confirmation. Aborting.")
            raise ValueError("Confirmation not provided to clear index. Set are_you_sure=True.")

        logger.warning(f"Attempting to clear index '{self.settings.INDEX_NAME}' of type '{self.settings.VECTOR_STORE_TYPE}'")
        
        self._index = None 

        if self.settings.VECTOR_STORE_TYPE == "simple":
            storage_path = Path("./storage") # Default LlamaIndex persistence path
            if hasattr(self._storage_context, '_persist_dir'): # If persist_dir was customized
                 storage_path = Path(self._storage_context._persist_dir)
            
            if storage_path.exists() and storage_path.is_dir():
                logger.info(f"Deleting SimpleVectorStore persistence directory: {storage_path}")
                shutil.rmtree(storage_path)
            self._initialize_components() # Re-initializes with fresh LlamaSettings and empty store
            logger.info("SimpleVectorStore cleared and RAGManager re-initialized.")

        elif self.settings.VECTOR_STORE_TYPE == "pinecone":
            try:
                import pinecone
                if not self.settings.PINECONE_API_KEY or not self.settings.PINECONE_ENVIRONMENT:
                    raise ConfigurationError("Pinecone API key or environment not set for clearing.")
                pinecone.init(api_key=self.settings.PINECONE_API_KEY, environment=self.settings.PINECONE_ENVIRONMENT)
                
                pinecone_index_name = self.settings.INDEX_NAME
                if hasattr(self._vector_store_llama_instance, 'pinecone_index_name'): # If store has specific name
                    pinecone_index_name = self._vector_store_llama_instance.pinecone_index_name
                
                if pinecone_index_name in pinecone.list_indexes():
                    logger.info(f"Deleting Pinecone index: {pinecone_index_name}")
                    pinecone.delete_index(pinecone_index_name)
                    logger.info(f"Pinecone index {pinecone_index_name} deleted.")
                else:
                    logger.info(f"Pinecone index {pinecone_index_name} not found, nothing to delete.")
                self._initialize_components() # Re-initializes, which might re-create the index structure if configured
            except Exception as e:
                logger.error(f"Error clearing Pinecone index: {e}", exc_info=True)
                raise IndexingError(f"Failed to clear Pinecone index: {e}")

        elif self.settings.VECTOR_STORE_TYPE == "postgres":
            logger.warning("Clearing PGVectorStore typically requires manual DB table TRUNCATE/DELETE. Re-initializing components will connect to the existing (possibly non-empty) table or create if not exists.")
            # For a true clear, you'd need direct DB operations:
            # try:
            #     pg_store = self._vector_store_llama_instance
            #     if pg_store and hasattr(pg_store, '_delete_index_data_from_table'):
            #         pg_store._delete_index_data_from_table() # Hypothetical method
            #         logger.info(f"Cleared data from PGVectorStore table: {pg_store.table_name}")
            #     else:
            #         # Connect to DB and TRUNCATE pg_store.table_name
            #         pass
            # except Exception as e:
            #    logger.error(f"Error clearing PGVectorStore data: {e}", exc_info=True)
            self._initialize_components() # Re-initializes connection
            logger.info("PGVectorStore RAGManager components re-initialized.")
        else:
            logger.error(f"Unsupported vector store type for clearing: {self.settings.VECTOR_STORE_TYPE}")
            raise ConfigurationError(f"Cannot clear index for type: {self.settings.VECTOR_STORE_TYPE}")
        
        logger.info("Index clear operation finished and RAGManager re-initialized.")