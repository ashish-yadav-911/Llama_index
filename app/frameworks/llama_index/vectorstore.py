# AGENTIC_MIRAI/app/frameworks/llama_index/vector_stores.py
from typing import Optional
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
from app.abstract.vectorstore import BaseVectorStore
from shared.config import Settings
from shared.exceptions import ConfigurationError
from shared.log import get_logger
import pinecone # type : warning # Import specific to pinecone

logger = get_logger(__name__)

class LlamaIndexVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._vector_store_instance: Optional[VectorStore] = None

    def get_vector_store(self) -> VectorStore:
        if self._vector_store_instance:
            return self._vector_store_instance

        store_type = self.settings.VECTOR_STORE_TYPE
        logger.info(f"Initializing LlamaIndex vector store: {store_type}")

        if store_type == "simple":
            self._vector_store_instance = SimpleVectorStore()
        elif store_type == "pinecone":
            if not self.settings.PINECONE_API_KEY or not self.settings.PINECONE_ENVIRONMENT:
                raise ConfigurationError("Pinecone API key or environment not set.")
            
            pinecone.init(api_key=self.settings.PINECONE_API_KEY, environment=self.settings.PINECONE_ENVIRONMENT)
            # Check if index exists, create if not
            # This logic can be more robust (e.g., check dimensions)
            if self.settings.INDEX_NAME not in pinecone.list_indexes():
                # You need to know the embedding dimension here.
                # This is a common chicken-and-egg problem.
                # For now, let's assume a common dimension or fetch it from a dummy embedding.
                # A better way is to get it from the embedder factory.
                # For BGE-Small-EN-v1.5 it's 384. For text-embedding-ada-002 it's 1536.
                # This should be configured or dynamically determined.
                # Let's hardcode for now or make it configurable
                # This is a simplification. In a real system, dimension should be managed carefully.
                if self.settings.EMBEDDING_MODEL_TYPE == "huggingface" and "bge-small" in self.settings.HF_EMBEDDING_MODEL_NAME:
                    dimension = 384
                elif self.settings.EMBEDDING_MODEL_TYPE == "openai" and "ada-002" in self.settings.OPENAI_EMBEDDING_MODEL_NAME:
                    dimension = 1536
                else: # A default, or raise error
                    logger.warning("Cannot determine embedding dimension for Pinecone index creation. Using default 384.")
                    dimension = 384 # Default or raise error
                
                pinecone.create_index(
                    self.settings.INDEX_NAME, 
                    dimension=dimension, # Example dimension, must match your embedding model
                    metric="cosine" # Or "euclidean", "dotproduct"
                )
                logger.info(f"Created Pinecone index: {self.settings.INDEX_NAME} with dimension {dimension}")
            
            pinecone_index = pinecone.Index(self.settings.INDEX_NAME)
            self._vector_store_instance = PineconeVectorStore(pinecone_index=pinecone_index)

        elif store_type == "postgres":
            if not all([self.settings.PG_HOST, self.settings.PG_USER, self.settings.PG_PASSWORD, self.settings.PG_DB_NAME]):
                raise ConfigurationError("PostgreSQL connection details not fully set.")
            
            # Construct connection string
            # Note: For async, use 'postgresql+asyncpg://'
            connection_string = f"postgresql://{self.settings.PG_USER}:{self.settings.PG_PASSWORD}@{self.settings.PG_HOST}:{self.settings.PG_PORT or 5432}/{self.settings.PG_DB_NAME}"
            
            # You might need to determine embed_dim similar to Pinecone
            # For BGE-Small-EN-v1.5 it's 384. For text-embedding-ada-002 it's 1536.
            if self.settings.EMBEDDING_MODEL_TYPE == "huggingface" and "bge-small" in self.settings.HF_EMBEDDING_MODEL_NAME:
                dimension = 384
            elif self.settings.EMBEDDING_MODEL_TYPE == "openai" and "ada-002" in self.settings.OPENAI_EMBEDDING_MODEL_NAME:
                dimension = 1536
            else: 
                logger.warning("Cannot determine embedding dimension for PGVector. Using default 384.")
                dimension = 384

            self._vector_store_instance = PGVectorStore.from_params(
                database=self.settings.PG_DB_NAME,
                host=self.settings.PG_HOST,
                password=self.settings.PG_PASSWORD,
                port=self.settings.PG_PORT or 5432,
                user=self.settings.PG_USER,
                table_name=self.settings.PG_TABLE_NAME,
                embed_dim=dimension # Must match your embedding model
            )
            # PGVectorStore might try to create the table, ensure DB user has permissions
            # And the 'vector' extension is enabled in PostgreSQL: CREATE EXTENSION vector;
            logger.info(f"Using PGVectorStore with table: {self.settings.PG_TABLE_NAME}")
        else:
            logger.error(f"Unsupported vector store type: {store_type}")
            raise ConfigurationError(f"Unsupported vector store type: {store_type}")
        return self._vector_store_instance

    def get_storage_context(self) -> StorageContext:
        vector_store = self.get_vector_store()
        return StorageContext.from_defaults(vector_store=vector_store)