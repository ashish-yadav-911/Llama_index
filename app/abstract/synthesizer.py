# # AGENTIC_MIRAI/app/abstract/synthesizer.py
# from abc import ABC, abstractmethod
# from typing import List, Optional
# from llama_index.core.schema import BaseNode # Or use a more generic type if needed

# class BaseResponseSynthesizer(ABC):
#     @abstractmethod
#     async def synthesize(
#         self, 
#         query: str, 
#         context_nodes: List[BaseNode], # LlamaIndex nodes
#         # prompt_template_str: Optional[str] = None # Prompt managed internally for now
#     ) -> str:
#         """
#         Synthesizes a response from the LLM based on the query and context.
#         """
#         pass

# AGENTIC_MIRAI/app/abstract/synthesizer.py
from abc import ABC, abstractmethod
from typing import List, Optional, Union # Ensure Union is here if you used it
from llama_index.core.schema import BaseNode, NodeWithScore # Or whatever types you settled on

class BaseResponseSynthesizer(ABC):
    @abstractmethod
    async def synthesize(
        self,
        query: str,
        context_items: List[Union[NodeWithScore, BaseNode]], # Or List[NodeWithScore]
    ) -> str:
        pass