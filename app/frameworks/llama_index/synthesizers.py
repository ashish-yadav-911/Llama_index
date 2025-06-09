# AGENTIC_MIRAI/app/frameworks/llama_index/synthesizers.py
import os
from typing import List, Optional, Union # Make sure Union is imported

from llama_index.core.llms import LLM
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode # Ensure TextNode and BaseNode are imported

from app.abstract.synthesizer import BaseResponseSynthesizer
from shared.config import Settings
from shared.log import get_logger
from shared.exceptions import ConfigurationError

logger = get_logger(__name__)

class LlamaIndexResponseSynthesizer(BaseResponseSynthesizer):
    def __init__(self, llm: LLM, settings: Settings):
        super().__init__() # Good practice
        self.llm = llm
        self.settings = settings
        self.prompt_template_str = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        prompt_file_path = self.settings.SYNTHESIS_PROMPT_FILE
        if prompt_file_path and os.path.exists(prompt_file_path):
            try:
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    loaded_prompt = f.read()
                logger.info(f"Successfully loaded synthesis prompt from: {prompt_file_path}")
                return loaded_prompt
            except Exception as e:
                logger.error(f"Error loading prompt from {prompt_file_path}: {e}. Falling back to default.")
        else:
            if prompt_file_path:
                logger.warning(f"Prompt file not found at {prompt_file_path}. Falling back to default prompt.")
            else:
                logger.info("No synthesis prompt file specified. Using default prompt.")
        return self.settings.DEFAULT_SYNTHESIS_PROMPT

    async def synthesize(
        self,
        query: str,
        # The input from RAGManager should be List[NodeWithScore]
        # Let's rename the parameter here to reflect what we expect from the RAGManager
        retrieved_nodes_with_scores: List[NodeWithScore],
    ) -> str:
        logger.info(f"Synthesizing response for query: '{query}' with {len(retrieved_nodes_with_scores)} retrieved items.")

        nodes_for_synthesis: List[BaseNode] = []
        if retrieved_nodes_with_scores:
            for item in retrieved_nodes_with_scores:
                # We strictly expect NodeWithScore here from the retriever.
                # If 'item' is not NodeWithScore, then the retriever returned something unexpected.
                if isinstance(item, NodeWithScore):
                    nodes_for_synthesis.append(item.node)
                elif isinstance(item, BaseNode): # Fallback / Defensive: should ideally not happen if retriever is consistent
                    logger.warning(f"Synthesizer received a BaseNode directly instead of NodeWithScore: {type(item)}. Using it directly.")
                    nodes_for_synthesis.append(item)
                else:
                    logger.error(f"CRITICAL: Synthesizer received an unexpected item type in retrieved_nodes_with_scores: {type(item)}. Skipping.")
                    # This is a more serious issue if it happens.
        else:
            logger.warning("No context items provided for synthesis. Synthesizer will proceed with empty node list.")

        try:
            text_qa_template = PromptTemplate(self.prompt_template_str)
            llama_index_synthesizer_instance = get_response_synthesizer(
                llm=self.llm,
                text_qa_template=text_qa_template,
            )
            
            logger.debug(f"Using prompt template for synthesis:\n{self.prompt_template_str[:500]}...")

            response_obj = await llama_index_synthesizer_instance.asynthesize(
                query=query,
                nodes=nodes_for_synthesis # This must be a List[BaseNode]
            )
            
            answer = str(response_obj) 
            logger.info("Successfully synthesized response.")
            return answer
        except AttributeError as ae:
            logger.error(f"AttributeError during LlamaIndex synthesis, often due to unexpected node structure: {ae}", exc_info=True)
            raise ConfigurationError(f"Synthesis failed due to AttributeError: {str(ae)}") # Re-raise
        except Exception as e:
            logger.error(f"Error during LlamaIndex response synthesis: {e}", exc_info=True)
            raise ConfigurationError(f"Synthesis failed: {str(e)}") # Re-raise


# # AGENTIC_MIRAI/app/frameworks/llama_index/synthesizers.py
# import os
# from typing import List, Optional, Union # Add Union

# from llama_index.core.llms import LLM
# from llama_index.core.response_synthesizers import (
#     get_response_synthesizer,
# )
# from llama_index.core.prompts import PromptTemplate
# from llama_index.core.schema import BaseNode, NodeWithScore, TextNode # Import TextNode for isinstance check
# from app.abstract.synthesizer import BaseResponseSynthesizer
# from shared.config import Settings
# from shared.log import get_logger
# from shared.exceptions import ConfigurationError

# logger = get_logger(__name__)

# class LlamaIndexResponseSynthesizer(BaseResponseSynthesizer):
#     def __init__(self, llm: LLM, settings: Settings):
#         self.llm = llm
#         self.settings = settings
#         self.prompt_template_str = self._load_prompt_template()

#     def _load_prompt_template(self) -> str:
#         prompt_file_path = self.settings.SYNTHESIS_PROMPT_FILE
#         if prompt_file_path and os.path.exists(prompt_file_path):
#             try:
#                 with open(prompt_file_path, 'r', encoding='utf-8') as f:
#                     loaded_prompt = f.read()
#                 logger.info(f"Successfully loaded synthesis prompt from: {prompt_file_path}")
#                 return loaded_prompt
#             except Exception as e:
#                 logger.error(f"Error loading prompt from {prompt_file_path}: {e}. Falling back to default.")
#         else:
#             if prompt_file_path: # Path was given but not found
#                 logger.warning(f"Prompt file not found at {prompt_file_path}. Falling back to default prompt.")
#             else: # No path was given
#                 logger.info("No synthesis prompt file specified. Using default prompt.")
        
#         return self.settings.DEFAULT_SYNTHESIS_PROMPT

#     async def synthesize(
#         self,
#         query: str,
#         # Allow context_nodes_with_scores to potentially contain a mix, or just BaseNodes if a component upstream changed it
#         context_items: List[Union[NodeWithScore, BaseNode]], # More flexible input type hint
#     ) -> str:
#         logger.info(f"Synthesizing response for query: '{query}' with {len(context_items)} context items.")

#         nodes_for_synthesis: List[BaseNode] = []
#         if context_items:
#             for item in context_items:
#                 if isinstance(item, NodeWithScore):
#                     nodes_for_synthesis.append(item.node)
#                 elif isinstance(item, BaseNode): # It's already a BaseNode (e.g., TextNode)
#                     nodes_for_synthesis.append(item)
#                 else:
#                     logger.warning(f"Unexpected item type in context_items: {type(item)}. Skipping.")
#         else:
#             logger.warning("No context items provided for synthesis.")
#             # No change here, nodes_for_synthesis remains empty, which is handled by the prompt.

#         try:
#             text_qa_template = PromptTemplate(self.prompt_template_str)
#             llama_index_synthesizer = get_response_synthesizer(
#                 llm=self.llm,
#                 text_qa_template=text_qa_template,
#             )
            
#             logger.debug(f"Using prompt template for synthesis:\n{self.prompt_template_str[:500]}...")

#             response_obj = await llama_index_synthesizer.asynthesize(
#                 query=query,
#                 nodes=nodes_for_synthesis # This now correctly contains only BaseNode objects
#             )
            
#             answer = str(response_obj) 
#             logger.info("Successfully synthesized response.")
#             return answer
#         except AttributeError as ae: # Catch AttributeError specifically to see if it's our 'node' problem
#             logger.error(f"AttributeError during synthesis, possibly related to node structure: {ae}", exc_info=True)
#             raise ConfigurationError(f"Synthesis failed due to AttributeError: {str(ae)}")
#         except Exception as e:
#             logger.error(f"Error during LlamaIndex response synthesis: {e}", exc_info=True)
#             raise ConfigurationError(f"Synthesis failed: {str(e)}")