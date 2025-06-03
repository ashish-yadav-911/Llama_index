# AGENTIC_MIRAI/tests/test_rag_manager_simple.py
import pytest
import os
from pathlib import Path
import shutil

from shared.config import Settings
from app.factory.component_factory import ComponentFactory
from app.modules.rag_manager import RAGManager
from shared.validation.query_schema import QueryRequest

# Create a temporary test_docs directory for this test
TEST_DOCS_DIR = Path(__file__).parent / "temp_test_docs"
SAMPLE_TEXT_CONTENT = "This is a test sentence about apples. Another sentence about bananas."

@pytest.fixture(scope="module") # Use module scope to setup/teardown once per test file
def test_settings(monkeypatch_module):
    # Override settings for testing, especially VECTOR_STORE_TYPE to "simple"
    monkeypatch_module.setenv("VECTOR_STORE_TYPE", "simple")
    monkeypatch_module.setenv("EMBEDDING_MODEL_TYPE", "huggingface") # Uses a local model
    monkeypatch_module.setenv("HF_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2") # A small fast one
    monkeypatch_module.setenv("LOG_LEVEL", "DEBUG") # More logs for tests
    # Ensure Pinecone/Postgres keys are not required if not used
    monkeypatch_module.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch_module.delenv("PINECONE_ENVIRONMENT", raising=False)
    # ... and for PG
    return Settings()

@pytest.fixture(scope="module")
def rag_manager_fixture(test_settings):
    # Clean up any previous simple vector store persistence
    storage_path = Path("./storage")
    if storage_path.exists():
        shutil.rmtree(storage_path)

    if TEST_DOCS_DIR.exists():
        shutil.rmtree(TEST_DOCS_DIR)
    TEST_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(TEST_DOCS_DIR / "sample_test.txt", "w") as f:
        f.write(SAMPLE_TEXT_CONTENT)

    factory = ComponentFactory(settings=test_settings)
    manager = RAGManager(settings=test_settings, component_factory=factory)
    
    yield manager # Provide the RAGManager instance

    # Teardown: Clean up test files and simple vector store persistence
    if TEST_DOCS_DIR.exists():
        shutil.rmtree(TEST_DOCS_DIR)
    if storage_path.exists(): # In case manager created it
        shutil.rmtree(storage_path)


def test_add_document_simple(rag_manager_fixture: RAGManager):
    test_file_path = str(TEST_DOCS_DIR / "sample_test.txt")
    nodes_indexed = rag_manager_fixture.add_document(
        file_path=test_file_path,
        chunk_size=50, # Small chunk for testing
        chunk_overlap=5
    )
    assert nodes_indexed > 0

def test_query_simple(rag_manager_fixture: RAGManager):
    # Ensure a document is added first if tests run independently or manager is reset
    # For this fixture scope, add_document in previous test should persist
    # If not, add it here:
    # test_file_path = str(TEST_DOCS_DIR / "sample_test.txt")
    # if not rag_manager_fixture._index or not rag_manager_fixture._index.docstore.docs: # crude check
    #     rag_manager_fixture.add_document(test_file_path, chunk_size=50, chunk_overlap=5)


    query_req = QueryRequest(query_text="apples", top_k=1)
    results = rag_manager_fixture.query(query_req)
    
    assert len(results) > 0
    assert "apples" in results[0].text.lower()
    assert results[0].score is not None
    assert results[0].score > 0.1 # Some arbitrary low similarity score check

    query_req_banana = QueryRequest(query_text="bananas", top_k=1)
    results_banana = rag_manager_fixture.query(query_req_banana)
    assert len(results_banana) > 0
    assert "bananas" in results_banana[0].text.lower()

def test_query_non_existent(rag_manager_fixture: RAGManager):
    query_req = QueryRequest(query_text="oranges a fruit not in doc", top_k=1)
    results = rag_manager_fixture.query(query_req)
    # Depending on retriever behavior, it might return less relevant docs or empty
    # If it returns docs, their scores should be lower.
    if results:
         assert "oranges" not in results[0].text.lower()
    # Or assert len(results) == 0 if strict no-match behavior is expected (unlikely for dense retrieval)

def test_clear_index_simple(rag_manager_fixture: RAGManager, test_settings: Settings):
    # Add a doc to ensure something is there
    test_file_path = str(TEST_DOCS_DIR / "sample_test.txt")
    rag_manager_fixture.add_document(test_file_path, chunk_size=30, chunk_overlap=5)

    query_req = QueryRequest(query_text="apples", top_k=1)
    results_before_clear = rag_manager_fixture.query(query_req)
    assert len(results_before_clear) > 0

    # Clear the index
    rag_manager_fixture.clear_index(are_you_sure=True)
    
    # Query again, should find nothing or manager should raise if index is truly gone
    # The RAGManager re-initializes an empty index after clearing.
    results_after_clear = rag_manager_fixture.query(query_req)
    assert len(results_after_clear) == 0 # Expect no results from an empty index

    # Verify storage path is gone if it was created (for simple store)
    storage_path = Path("./storage") # Default LlamaIndex persistence for SimpleVectorStore
    assert not storage_path.exists() # clear_index for simple store should remove this

# To run this specific test:
# pytest tests/test_rag_manager_simple.py