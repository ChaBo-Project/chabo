import os
import configparser
import logging
logger = logging.getLogger(__name__)
import requests # Need this for _call_hf_endpoint, but we will define the function here
import httpx
import json
from typing import Dict, Any, List
from langchain_core.documents import Document



def getconfig(configfile_path: str) -> configparser.ConfigParser:
    """Reads the config file."""
    config = configparser.ConfigParser()
    try:
        config.read_file(open(configfile_path))
        return config
    except FileNotFoundError:
        logger.error(f"Warning: Config file not found at {configfile_path}. Relying on environment variables.")
        return configparser.ConfigParser()


def get_auth_for_generator(provider: str) -> dict:
    """Get authentication configuration for different providers"""
    auth_configs = {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "huggingface": {"api_key": os.getenv("HF_TOKEN")},
        "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY")},
        "cohere": {"api_key": os.getenv("COHERE_API_KEY")},
    }
    
    if provider not in auth_configs:
        logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
    
    auth_config = auth_configs[provider]
    api_key = auth_config.get("api_key")
    
    if not api_key:
        logger.error(f"Missing API key for provider '{provider}'. Please set the appropriate environment variable.")
        raise RuntimeError(f"Missing API key for provider '{provider}'. Please set the appropriate environment variable.")
    
    return auth_config



def get_config_value(config: configparser.ConfigParser, section: str, key: str, env_var: str, fallback: Any = None) -> Any:
    """
    Retrieves a config value, prioritizing: Environment Variable > Config File > Fallback.
    """
    # 1. Check Environment Variable (Highest Priority)
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value
        
    # 2. Check Config File
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        # 3. Use Fallback
        if fallback is not None:
            return fallback
        
        # 4. Error if essential config is missing
        logger.error(f"Configuration missing: Required value for [{section}]{key} was not found ")
        logger.error(f"in 'params.cfg' and the environment variable {env_var} is not set.")
        raise ValueError(
            f"Configuration missing: Required value for [{section}]{key} was not found "
            f"in 'params.cfg' and the environment variable {env_var} is not set."
        )

def _call_hf_endpoint(url: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper for making authenticated requests to Hugging Face Endpoints."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        logger.info(f"Calling endpoint {url}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling HF endpoint ({url}): {e}")
        raise

async def _acall_hf_endpoint(url: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronously calls a Hugging Face Inference Endpoint using httpx."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # Use httpx.AsyncClient for asynchronous requests
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Async Calling endpoint {url}")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error calling HF endpoint ({url}): {e}")
            raise
