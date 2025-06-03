# AGENTIC_MIRAI/shared/exceptions.py
class BaseRAGException(Exception):
    """Base exception for the RAG system."""
    def __init__(self, message="An error occurred in the RAG system."):
        self.message = message
        super().__init__(self.message)

class ConfigurationError(BaseRAGException):
    """Exception for configuration-related errors."""
    def __init__(self, message="Configuration error."):
        super().__init__(message)

class DocumentProcessingError(BaseRAGException):
    """Exception for errors during document loading or processing."""
    def __init__(self, message="Error processing document."):
        super().__init__(message)

class IndexingError(BaseRAGException):
    """Exception for errors during vector store indexing."""
    def __init__(self, message="Error during indexing."):
        super().__init__(message)

class QueryError(BaseRAGException):
    """Exception for errors during querying."""
    def __init__(self, message="Error during query execution."):
        super().__init__(message)

class UnsupportedFeatureError(BaseRAGException):
    """Exception for requested features that are not supported or configured."""
    def __init__(self, message="Unsupported feature requested."):
        super().__init__(message)