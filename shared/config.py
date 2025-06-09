# AGENTIC_MIRAI/shared/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field, ValidationInfo
from typing import Literal, Optional, Any

# Default prompt if the file is not found or specified
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

    # ... existing log, chunking, embedding, vector_store settings ...
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_ROTATION_WHEN: Literal["S", "M", "H", "D", "midnight"] = "midnight"
    LOG_ROTATION_INTERVAL: int = 1
    LOG_ROTATION_BACKUP_COUNT: int = 30

    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50

    EMBEDDING_MODEL_TYPE: Literal["huggingface", "openai"] = "openai"
    HF_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"

    VECTOR_STORE_TYPE: Literal["simple", "pinecone", "postgres"] = "simple"
    INDEX_NAME: str = "my-rag-index" 
    PG_TABLE_NAME: str = "llama_index_vectors"

    DEFAULT_RETRIEVER_TOP_K: int = 3
    DEFAULT_RETRIEVER_STRATEGY: Literal["similarity", "mmr"] = "similarity"
    MMR_DIVERSITY_BIAS: float = 0.5

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # --- Keys and Credentials ---
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    PINECONE_API_KEY: Optional[str] = Field(None, description="Pinecone API Key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(None, description="Pinecone Environment")
    PG_HOST: Optional[str] = Field(None, description="PostgreSQL Host")
    PG_PORT: Optional[int] = Field(None, description="PostgreSQL Port")
    PG_USER: Optional[str] = Field(None, description="PostgreSQL User")
    PG_PASSWORD: Optional[str] = Field(None, description="PostgreSQL Password")
    PG_DB_NAME: Optional[str] = Field(None, description="PostgreSQL Database Name")

    # --- LLM Settings (for Synthesis) ---
    LLM_PROVIDER: Literal["openai"] = Field("openai", description="LLM Provider for synthesis") # Default to openai
    OPENAI_LLM_MODEL_NAME: str = Field("gpt-4o-mini", description="OpenAI model name for synthesis")
    OPENAI_LLM_TEMPERATURE: float = Field(0.1, ge=0.0, le=2.0, description="Temperature for LLM generation")
    SYNTHESIS_PROMPT_FILE: Optional[str] = Field("prompts/default_synthesis_prompt.txt", description="Path to the synthesis prompt template file")
    DEFAULT_SYNTHESIS_PROMPT: str = Field(DEFAULT_SYNTHESIS_PROMPT_TEMPLATE_STR, description="Default synthesis prompt string if file is not used/found")


    @field_validator("OPENAI_API_KEY", mode='before')
    @classmethod
    def check_openai_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if (info.data.get("EMBEDDING_MODEL_TYPE") == "openai" or \
            info.data.get("LLM_PROVIDER") == "openai"): # Check if LLM_PROVIDER is also openai
            if not v:
                raise ValueError("OPENAI_API_KEY must be set if using OpenAI embeddings or LLM.")
        return v

    @field_validator("PINECONE_API_KEY", "PINECONE_ENVIRONMENT", mode='before')
    @classmethod
    def check_pinecone_creds(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if info.data.get("VECTOR_STORE_TYPE") == "pinecone":
            if not v:
                raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'pinecone'.")
        return v
        
    @field_validator("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DB_NAME", mode='before')
    @classmethod
    def check_required_postgres_creds(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if info.data.get("VECTOR_STORE_TYPE") == "postgres":
            if not v: # These specific fields must not be None
                raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'postgres'.")
        return v

    @field_validator("PG_PORT", mode='before')
    @classmethod
    def check_postgres_port(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        # PG_PORT can be None (to use default), so no specific "must be set" validation here for this field alone
        # if VECTOR_STORE_TYPE is "postgres" and it's None, it's acceptable for this field.
        # Type validation (must be int if provided) is handled by Pydantic.
        return v

settings = Settings()