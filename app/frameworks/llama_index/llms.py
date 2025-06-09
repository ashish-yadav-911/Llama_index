# # AGENTIC_MIRAI/app/frameworks/llama_index/llms.py
# from llama_index.core.llms import LLM
# from llama_index.llms.openai import OpenAI # Specific import for OpenAI
# # from llama_index.llms.huggingface import HuggingFaceLLM # Example for future

# from shared.config import Settings
# from shared.exceptions import ConfigurationError
# from shared.log import get_logger

# logger = get_logger(__name__)

# def get_llama_index_llm(settings: Settings) -> LLM:
#     """
#     Factory function to create and return a LlamaIndex LLM instance
#     based on the application settings.
#     """
#     llm_provider = settings.LLM_PROVIDER
#     logger.info(f"Initializing LLM for provider: {llm_provider}")

#     if llm_provider == "openai":
#         if not settings.OPENAI_API_KEY:
#             raise ConfigurationError("OPENAI_API_KEY is not set for OpenAI LLM.")
        
#         logger.debug(f"OpenAI LLM params: model={settings.OPENAI_LLM_MODEL_NAME}, temp={settings.OPENAI_LLM_TEMPERATURE}")
        
#         try:
#             return OpenAI(
#                 model=settings.OPENAI_LLM_MODEL_NAME,
#                 api_key=settings.OPENAI_API_KEY,
#                 temperature=settings.OPENAI_LLM_TEMPERATURE,
#                 logprobs=False,  # <<< ADDED: Explicitly set (usually defaults to False)
#                 # default_headers={}, # <<< OPTION 1: Pass empty dict (Try this first)
#                 # You can add other parameters like max_tokens, etc.
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
#             # Re-raise as ConfigurationError or a more specific LLMInitializationError
#             raise ConfigurationError(f"Failed to initialize OpenAI LLM: {str(e)}")
            
#     # Add other providers here later
#     else:
#         logger.error(f"Unsupported LLM provider: {llm_provider}")
#         raise ConfigurationError(f"Unsupported LLM provider: {llm_provider}")

# AGENTIC_MIRAI/app/frameworks/llama_index/llms.py
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI # Specific import for OpenAI
# from llama_index.llms.huggingface import HuggingFaceLLM # Example for future

from shared.config import Settings
from shared.exceptions import ConfigurationError
from shared.log import get_logger

logger = get_logger(__name__)

def get_llama_index_llm(settings: Settings) -> LLM:
    """
    Factory function to create and return a LlamaIndex LLM instance
    based on the application settings.
    """
    llm_provider = settings.LLM_PROVIDER
    logger.info(f"Initializing LLM for provider: {llm_provider}")

    if llm_provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ConfigurationError("OPENAI_API_KEY is not set for OpenAI LLM.")
        
        logger.debug(f"OpenAI LLM params: model={settings.OPENAI_LLM_MODEL_NAME}, temp={settings.OPENAI_LLM_TEMPERATURE}")
        
        try:
            return OpenAI(
                model=settings.OPENAI_LLM_MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.OPENAI_LLM_TEMPERATURE,
                logprobs=False,
                default_headers={}, # <<< ENSURE THIS LINE IS PRESENT AND UNCOMMENTED
                # You can add other parameters like max_tokens, etc.
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize OpenAI LLM: {str(e)}")
            
    else:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        raise ConfigurationError(f"Unsupported LLM provider: {llm_provider}")