# AGENTIC_MIRAI/shared/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_ROTATION_WHEN: Literal["S", "M", "H", "D", "midnight"] = "midnight"
    LOG_ROTATION_INTERVAL: int = 1
    LOG_ROTATION_BACKUP_COUNT: int = 30

    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 50

    EMBEDDING_MODEL_TYPE: Literal["huggingface", "openai"] = "huggingface"
    HF_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"

    VECTOR_STORE_TYPE: Literal["simple", "pinecone", "postgres"] = "simple"
    INDEX_NAME: str = "my-rag-index"

    PINECONE_API_KEY: Optional[str] = Field(None, description="Pinecone API Key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(None, description="Pinecone Environment")

    PG_HOST: Optional[str] = Field(None, description="PostgreSQL Host")
    PG_PORT: Optional[int] = Field(None, description="PostgreSQL Port")
    PG_USER: Optional[str] = Field(None, description="PostgreSQL User")
    PG_PASSWORD: Optional[str] = Field(None, description="PostgreSQL Password")
    PG_DB_NAME: Optional[str] = Field(None, description="PostgreSQL Database Name")
    PG_TABLE_NAME: str = "llama_index_vectors"

    DEFAULT_RETRIEVER_TOP_K: int = 3
    DEFAULT_RETRIEVER_STRATEGY: Literal["similarity", "mmr"] = "similarity"
    MMR_DIVERSITY_BIAS: float = 0.5

    LLM_PROVIDER: Optional[Literal["openai"]] = Field(None, description="LLM Provider for synthesis")
    OPENAI_LLM_MODEL_NAME: str = "gpt-3.5-turbo"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @field_validator("OPENAI_API_KEY", mode="before")
    @classmethod
    def check_openai_key(cls, v, info):
        values = info.data
        if values.get("EMBEDDING_MODEL_TYPE") == "openai" or values.get("LLM_PROVIDER") == "openai":
            if not v:
                raise ValueError("OPENAI_API_KEY must be set if using OpenAI embeddings or LLM.")
        return v

    @field_validator("PINECONE_API_KEY", "PINECONE_ENVIRONMENT", mode="before")
    @classmethod
    def check_pinecone_creds(cls, v, info):
        values = info.data
        if values.get("VECTOR_STORE_TYPE") == "pinecone":
            if not v:
                raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'pinecone'.")
        return v

    @field_validator("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DB_NAME", "PG_PORT", mode="before")
    @classmethod
    def check_postgres_creds(cls, v, info):
        values = info.data
        if values.get("VECTOR_STORE_TYPE") == "postgres":
            if not v:
                if info.field_name != "PG_PORT" or (info.field_name == "PG_PORT" and v is None):
                    raise ValueError(f"{info.field_name} must be set if VECTOR_STORE_TYPE is 'postgres'.")
        return v


settings = Settings()
