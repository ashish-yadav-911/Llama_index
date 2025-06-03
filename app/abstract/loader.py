# AGENTIC_MIRAI/app/abstract/loader.py
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
from llama_index.core.schema import Document # type: ignore


class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, source: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Loads documents from a source."""
        pass