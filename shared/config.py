# AGENTIC_MIRAI/shared/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field, ValidationInfo
from typing import Literal, Optional, Any

# This constant is fine outside the class
DEFAULT_SYNTHESIS_PROMPT_TEMPLATE_STR = """
You are a helpful personal assistant for document analysis.
Based on the following context information, please answer the user's query.
The context consists of several text chunks, which may or may not be relevant.
Synthesize a concise and informative answer.
If the context does not provide enough information to answer the query, please state that you cannot answer based on the provided documents.
Do not make up information or refer to prior knowledge outside of the context.

Context information is below:
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- API Server Settings ---
    API_HOST: str = Field("0.0.0.0", description="Host for the API server")
    API_PORT: int = Field(8000, description="Port for the API server")

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FILE_PATH: str = Field("logs/app.log", description="Path to the log file")
    LOG_ROTATION_WHEN: Literal["S", "M", "H", "D", "midnight"] = Field("midnight", description="Log rotation time unit")
    LOG_ROTATION_INTERVAL: int = Field(1, description="Log rotation interval")
    LOG_ROTATION_BACKUP_COUNT: int = Field(30, description="Number of backup log files")

    # --- Document Processing & LlamaIndex Defaults ---
    DEFAULT_CHUNK_SIZE: int = Field(512, description="Default chunk size")
    DEFAULT_CHUNK_OVERLAP: int = Field(50, description="Default chunk overlap")
    INDEX_NAME: str = Field("my-rag-index", description="General name for the LlamaIndex index")

    # --- API Keys & External Service Credentials ---
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    PINECONE_API_KEY: Optional[str] = Field(None, description="Pinecone API Key")
    # Add other global API keys here, e.g., ANTHROPIC_API_KEY

    # --- Embedding Model Configuration ---
    EMBEDDING_MODEL_TYPE: Literal["huggingface", "openai"] = Field("openai", description="Embedding provider")
    HF_EMBEDDING_MODEL_NAME: str = Field("BAAI/bge-small-en-v1.5", description="HuggingFace embedding model")
    OPENAI_EMBEDDING_MODEL_NAME: str = Field("text-embedding-3-large", description="OpenAI embedding model") 
    
    # --- LLM (Synthesis) Configuration ---
    LLM_PROVIDER: Literal["openai"] = Field("openai", description="LLM provider for synthesis")
    OPENAI_LLM_MODEL_NAME: str = Field("gpt-4o-mini", description="OpenAI LLM for synthesis")
    OPENAI_LLM_TEMPERATURE: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature")
    # DEFAULT_SYNTHESIS_PROMPT: str = Field(DEFAULT_SYNTHESIS_PROMPT_TEMPLATE_STR, description="Default synthesis prompt string if building custom synthesizer")

    # --- Vector Store Configuration ---
    VECTOR_STORE_TYPE: Literal["simple", "pinecone", "postgres"] = Field("simple", description="Vector store type")

    # --- Pinecone Specific Settings ---
    PINECONE_ENVIRONMENT: Optional[str] = Field(None, description="Pinecone Environment (optional if host is used primarily)")
    PINECONE_INDEX_HOST: Optional[str] = Field(None, description="Pinecone Index Host URL")
    PINECONE_NAMESPACE: Optional[str] = Field(None, description="Pinecone namespace")

    # --- PostgreSQL (PGVector) Specific Settings ---
    PG_HOST: Optional[str] = Field(None, description="PostgreSQL Host")
    PG_PORT: Optional[int] = Field(None, description="PostgreSQL Port (defaults if None)")
    PG_USER: Optional[str] = Field(None, description="PostgreSQL User")
    PG_PASSWORD: Optional[str] = Field(None, description="PostgreSQL Password")
    PG_DB_NAME: Optional[str] = Field(None, description="PostgreSQL Database Name")
    PG_TABLE_NAME: str = Field("llama_index_vectors", description="PGVector table name")

    # --- Retriever Configuration ---
    DEFAULT_RETRIEVER_TOP_K: int = Field(3, description="Default K for retrieval")
    DEFAULT_RETRIEVER_STRATEGY: Literal["similarity", "mmr", "hybrid"] = Field("similarity", description="Default retrieval strategy")
    DEFAULT_MMR_THRESHOLD: float = Field(0.5, ge=0.0, le=1.0, description="Default MMR threshold")
    # Defaults for hybrid search components are usually handled within the retriever logic itself or could be added here if highly configurable.
    # DEFAULT_BM25_TOP_K: int = Field(5)
    # DEFAULT_QUERY_FUSION_SIMILARITY_TOP_K: int = Field(3)


    # --- Validators ---

    @field_validator("OPENAI_API_KEY", mode='before')
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # This validator correctly checks if OPENAI_API_KEY is needed based on other settings
        config_data = info.data # Get the dictionary of all field values being processed
        if config_data.get("EMBEDDING_MODEL_TYPE") == "openai" or config_data.get("LLM_PROVIDER") == "openai":
            if not v: # v is the current value of OPENAI_API_KEY
                raise ValueError("OPENAI_API_KEY must be set if using OpenAI for embeddings or as LLM provider.")
        return v

    @field_validator("PINECONE_API_KEY", "PINECONE_INDEX_HOST", mode='before')
    @classmethod
    def validate_pinecone_settings(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # Validates that PINECONE_API_KEY and PINECONE_INDEX_HOST are set if VECTOR_STORE_TYPE is "pinecone".
        # PINECONE_ENVIRONMENT is left as truly optional at the Pydantic level.
        config_data = info.data
        if config_data.get("VECTOR_STORE_TYPE") == "pinecone":
            # 'info.field_name' will be "PINECONE_API_KEY" or "PINECONE_INDEX_HOST" here.
            if not v: # v is the current value of the field being validated.
                raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'pinecone'.")
        return v
        
    @field_validator("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DB_NAME", mode='before')
    @classmethod
    def validate_postgres_credentials(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # Validates that specific PostgreSQL fields are set if VECTOR_STORE_TYPE is "postgres".
        # PG_PORT is optional (can be None to use default) and doesn't need this specific "must be set" check.
        config_data = info.data
        if config_data.get("VECTOR_STORE_TYPE") == "postgres":
            # 'info.field_name' will be one of "PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DB_NAME".
            if not v: # v is the current value of the field being validated.
                raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'postgres'.")
        return v

    # No specific validator is needed for PG_PORT's existence, as Optional[int] handles it.
    # If it's provided, Pydantic ensures it's an int. If not, it's None.

settings = Settings()