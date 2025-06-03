# AGENTIC_MIRAI/tests/test_config.py
import pytest
from shared.config import Settings
from pydantic import ValidationError

def test_load_settings(monkeypatch):
    # Example: Mock environment variables
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_CHUNK_SIZE", "256")
    
    settings = Settings() # This will re-read .env and then override with monkeypatch
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.DEFAULT_CHUNK_SIZE == 256

def test_openai_key_validation(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_TYPE", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False) # Ensure it's not set
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert "OPENAI_API_KEY must be set" in str(excinfo.value)

    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    settings = Settings()
    assert settings.OPENAI_API_KEY == "test_key"