import logging
# Set up logger
logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Union, AsyncGenerator
import asyncio

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel # For typing
from ..utils import getconfig, get_auth_for_generator, get_config_value
from .prompts import system_prompt, build_messages
from .sources import (
    process_context, parse_citations, 
    extract_sources, create_sources_list, clean_citations
)


class Generator:
    """
    A generic RAG answer generation component that supports multiple LLM providers 
    and reads configuration from kwargs, environment variables, or params.cfg.
    """
    
    # 1. Define the configuration map for all parameters
    CONFIG_MAP = {
        "provider":           ("generator", "PROVIDER", "GENERATOR_PROVIDER"),
        "model":              ("generator", "MODEL", "GENERATOR_MODEL"),
        "max_tokens":         ("generator", "MAX_TOKENS", "GENERATOR_MAX_TOKENS", 512),
        "temperature":        ("generator", "TEMPERATURE", "GENERATOR_TEMPERATURE", 0.1),
        "inference_provider": ("generator", "INFERENCE_PROVIDER", "GENERATOR_INFERENCE_PROVIDER"),
        "organization":       ("generator", "ORGANIZATION", "GENERATOR_ORGANIZATION"),

        # New: Metadata/RAG Configuration Parameters
        # Note: Fallbacks are strings that need to be parsed into lists later.
        "context_metadata_fields":  ("generator", "CONTEXT_META_FIELDS", "GENERATOR_CONTEXT_META_FIELDS", "source,page"),
        "title_metadata_fields":    ("generator", "TITLE_META_FIELDS", "GENERATOR_TITLE_META_FIELDS", "source,document_id"),
        "link_metadata_field":      ("generator", "LINK_META_FIELD", "GENERATOR_LINK_META_FIELD", "url")

    }
    
    def __init__(self, config_path: str = "params.cfg", **kwargs):
        logger.info("Initializing Generator component with config precedence...")
        
        # 2. Load Configuration
        config_file = getconfig(config_path)
        
        # Dictionary to store the resolved configuration
        resolved_config = {}
        
        for key, params in self.CONFIG_MAP.items():
            section, option, env_var = params[:3]
            fallback = params[3] if len(params) > 3 else None
            
            # 1. Prioritize kwargs (explicitly passed to the constructor)
            if key in kwargs:
                value = kwargs[key]
                logger.debug(f"Config '{key}' loaded from kwargs.")
            else:
                # 2. Use the unified utility to check ENV then config file
                value = get_config_value(config_file, section, option, env_var, fallback)
                
            # Type conversion
            if key in ['max_tokens']:
                value = int(value)
            elif key in ['temperature']:
                value = float(value)
            elif key in ['context_metadata_fields', 'title_metadata_fields']:
                # For list fields, parse the comma-separated string unless it's already a list from kwargs
                if isinstance(value, str):
                    value = [item.strip() for item in value.split(',') if item.strip()]
            
                
            resolved_config[key] = value
        

        # 3. Assign resolved values to instance attributes
        self.provider = resolved_config['provider']
        self.model = resolved_config['model']
        self.max_tokens = resolved_config['max_tokens']
        self.temperature = resolved_config['temperature']
        self.inference_provider = resolved_config.get('inference_provider')
        self.organization = resolved_config.get('organization')

        #  Assign resolved values to instance attributes (RAG/Metadata)
        self.context_metadata_fields = resolved_config['context_metadata_fields']
        self.title_metadata_fields = resolved_config['title_metadata_fields']
        self.link_metadata_field = resolved_config['link_metadata_field']

        # 4. Set up authentication using the resolved provider
        try:
            self.auth_config = get_auth_for_generator(self.provider)
        except Exception as e:
            logger.error(f"Failed to get authentication for provider '{self.provider}'. Ensure auth is configured.")
            raise e
        
        # 5. Initialize the Chat Model
        self.chat_model: BaseChatModel = self._get_chat_model()
        logger.info(f"Generator initialized with provider: {self.provider}, model: {self.model}")
        logger.debug(f"Metadata Config: Context={self.context_metadata_fields}, Title={self.title_metadata_fields}")

    def _get_chat_model(self) -> BaseChatModel:
        """Initialize the appropriate LangChain chat model based on provider."""
        common_params = {"temperature": self.temperature, "max_tokens": self.max_tokens, "streaming": True}
        
        providers = {
            "openai": lambda: ChatOpenAI(model=self.model, openai_api_key=self.auth_config["api_key"], **common_params),
            "anthropic": lambda: ChatAnthropic(model=self.model, anthropic_api_key=self.auth_config["api_key"], **common_params),
            "cohere": lambda: ChatCohere(model=self.model, cohere_api_key=self.auth_config["api_key"], **common_params),
            "huggingface": lambda: self._get_hf_chat_model(**common_params),
        }

        if self.provider not in providers:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return providers[self.provider]()

    def _get_hf_chat_model(self, **common_params) -> BaseChatModel:
        """Helper to configure HuggingFace model for Serverless or TGI Endpoint."""
        
        # Check if the MODEL config parameter looks like a URL (custom TGI endpoint)
        if self.model.startswith("http"):
            logger.info(f"Using TGI Hosted Endpoint: {self.model}")
            # --- TGI Endpoint Configuration ---
            llm = HuggingFaceEndpoint(
                endpoint_url=self.model,  # The model parameter is the TGI URL
                huggingfacehub_api_token=self.auth_config["api_key"], 
                # Note: Setting the task is often unnecessary for TGI endpoints
                task="text-generation", 
                temperature=self.temperature, 
                max_new_tokens=self.max_tokens, 
                streaming=True
            )
        else:
            logger.info(f"Using Hugging Face Serverless API for model: {self.model}")
            # --- Serverless Configuration ---
            llm = HuggingFaceEndpoint(
                repo_id=self.model, # The model parameter is the model ID
                huggingfacehub_api_token=self.auth_config["api_key"], 
                task="text-generation", 
                provider=self.inference_provider,
                server_kwargs={"bill_to": self.organization},
                temperature=self.temperature, 
                max_new_tokens=self.max_tokens, 
                streaming=True
            )
        
        # ChatHuggingFace wraps the endpoint and handles chat templating
        return ChatHuggingFace(llm=llm).bind(max_tokens = self.max_tokens)


    # # --- Internal LLM Call Functions ---

    async def _call_llm(self, messages: list) -> str:
        """Provider-agnostic LLM call using LangChain (non-streaming)"""
        try:
            # We use ainvoke as the class handles model init
            logger.debug("Calling LLM without streaming")
            response = await self.chat_model.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            raise

    async def _call_llm_streaming(self, messages: list) -> AsyncGenerator[str, None]:
        """Provider-agnostic streaming LLM call using LangChain"""
        try:
            # We use astream as the class handles model init
            logger.debug("Calling LLM with streaming")
            async for chunk in self.chat_model.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                
        except Exception as e:
            logger.exception(f"LLM streaming failed: {e}")
            # Stream the error message out to the consumer
            yield f"Error: {str(e)}"
    
    # # --- Response Post-Processing Method (Centralized Logic) ---

    # def _process_final_response(self, answer: str, processed_results: List[Dict[str, Any]], chatui_format: bool) -> Union[str, Dict[str, Any]]:
    #     """Handles final citation cleaning and output formatting."""
    #     # Clean citations
    #     answer = clean_citations(answer)

    #     if chatui_format:
    #         result = {"answer": answer}
    #         if processed_results:
    #             # Only include sources if processing successful
    #             cited_numbers = parse_citations(answer)
    #             cited_sources = extract_sources(processed_results, cited_numbers)
    #             result["sources"] = create_sources_list(cited_sources)
    #         return result
    #     else:
    #         return answer

    # # --- Main Generation Methods (Public Interface) ---

    async def generate(self, query: str, context: Union[str, List[Dict[str, Any]], None], chatui_format: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate an answer to a query using provided context (non-streaming)"""
        if not query.strip():
            error_msg = "Query cannot be empty"
            return {"error": error_msg} if chatui_format else f"Error: {error_msg}"
        logger.info(f"Generating answer for query: {query[:50]}")
        logger.info(f"CHATUI format is {chatui_format}")

        try:
            # 1. Process Context
            formatted_context, processed_results = process_context(context, 
               metadata_fields_to_include=self.context_metadata_fields)
            
            # 2. Build Messages (with system prompt)
            messages = build_messages(system_prompt, query, formatted_context)
            

            
            # 3. Call LLM
            answer = await self._call_llm(messages)
            
            if chatui_format:
                result = {"answer": answer}
                if processed_results:
                    cited_numbers = parse_citations(answer)
                    cited_sources = extract_sources(processed_results, cited_numbers)
                    result["sources"] = create_sources_list(cited_sources, 
                                        title_metadata_fields=self.title_metadata_fields,
                                        link_metadata_field=self.link_metadata_field)
                return result
            else:
                return answer       

        except Exception as e:
            logger.exception("Generation failed")
            error_msg = str(e)
            return {"error": error_msg} if chatui_format else f"Error: {error_msg}"

    async def generate_streaming(self, query: str, context: Union[str, List[Dict[str, Any]], None], chatui_format: bool = False) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate a streaming answer to a query using provided context through RAG"""
        if not query.strip():
            error_msg = "Query cannot be empty"
            # Return error in the streaming format
            if chatui_format:
                yield {"event": "error", "data": {"error": error_msg}}
            else:
                yield f"Error: {error_msg}"
            return
        logger.info(f"Generating streaming answer for query: {query[:50]}")
        logger.info(f"CHATUI format is {chatui_format}")

        try:
            # 1. Process Context
            formatted_context, processed_results = process_context(context,
                            metadata_fields_to_include =self.context_metadata_fields)
            
            # 2. Build Messages (with system prompt)
            messages = build_messages(system_prompt, query, formatted_context)

            # 3. Stream the response and accumulate for citation parsing
            accumulated_response = ""
            async for chunk in self._call_llm_streaming(messages):
                accumulated_response += chunk
                
                # Yield the raw text chunks immediately
                if chatui_format:
                    yield {"event": "data", "data": chunk}
                else:
                    yield chunk

            # 4. Final Post-Processing (after stream is complete)
            cleaned_response = clean_citations(accumulated_response)
            
            # Send sources at the end if available and in ChatUI format
            if chatui_format and processed_results:
                cited_numbers = parse_citations(cleaned_response)
                cited_sources = extract_sources(processed_results, cited_numbers)
                sources = create_sources_list(cited_sources, 
                            title_metadata_fields=self.title_metadata_fields,
                            link_metadata_field=self.link_metadata_field)
                logging.debug(f"Sorces recieved: {sources}")
                yield {"event": "sources", "data": {"sources": sources}}

            # Send END event for ChatUI format
            if chatui_format:
                yield {"event": "end", "data": {}}

        except Exception as e:
            logger.exception("Streaming generation failed")
            error_msg = str(e)
            if chatui_format:
                yield {"event": "error", "data": {"error": error_msg}}
            else:
                yield f"Error: {error_msg}"

