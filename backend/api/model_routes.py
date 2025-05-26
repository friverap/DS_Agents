from fastapi import APIRouter, HTTPException, Response, Depends, Body
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import os
from dotenv import load_dotenv, set_key
from pathlib import Path
from utils.model_utils import model_manager
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

class ModelConfig(BaseModel):
    provider: str
    model: str

class ProviderInfo(BaseModel):
    models: List[str]
    default_model: str

class ApiKeyUpdate(BaseModel):
    provider: str
    apiKey: str

@router.get("/providers")
async def get_available_providers() -> Dict[str, ProviderInfo]:
    """Get available AI providers and their models."""
    return model_manager.get_available_providers()

@router.get("/current")
async def get_current_model_config() -> ModelConfig:
    """Get the current model configuration."""
    return ModelConfig(
        provider=settings.DSPY_SETTINGS["provider"],
        model=settings.DSPY_SETTINGS["model"]
    )

@router.post("/configure")
async def configure_model(config: ModelConfig) -> Dict[str, bool]:
    """Configure the AI model to use."""
    success = model_manager.update_dspy_config(config.provider, config.model)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to update model configuration to {config.provider}/{config.model}"
        )
    
    return {"success": True}

@router.get("/api-keys/status")
async def get_api_key_status() -> Dict[str, bool]:
    """Check if API keys are configured for each provider."""
    status = {}
    
    for provider, config in settings.AI_PROVIDERS.items():
        if provider == "ollama":
            # Ollama is a local provider, check if the service is reachable
            status[provider] = provider in model_manager.clients
        else:
            # For cloud providers, check if we have an API key
            api_key_var = config.get("api_key_var")
            if api_key_var and getattr(settings, api_key_var):
                status[provider] = True
            else:
                status[provider] = False
    
    return status

@router.post("/api-keys/update")
async def update_api_key(update: ApiKeyUpdate) -> Dict[str, bool]:
    """Update API key for a specific provider."""
    provider = update.provider
    api_key = update.apiKey
    
    if provider not in settings.AI_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider}"
        )
    
    try:
        # Get the environment variable name for this provider's API key
        if provider == "ollama":
            api_var = "OLLAMA_API_BASE"
        else:
            api_var = settings.AI_PROVIDERS[provider].get("api_key_var")
        
        if not api_var:
            raise HTTPException(
                status_code=400,
                detail=f"No API key configuration available for {provider}"
            )
        
        # Update the environment variable
        os.environ[api_var] = api_key
        
        # Also try to update the .env file if it exists
        env_path = Path(settings.BASE_DIR.parent, '.env')
        if env_path.exists():
            # We manually update the .env file
            try:
                set_key(str(env_path), api_var, api_key)
                logger.info(f"Updated {api_var} in .env file")
            except Exception as e:
                logger.error(f"Failed to update .env file: {str(e)}")
        
        # Update settings module
        setattr(settings, api_var, api_key)
        
        # Reinitialize the client for this provider
        model_manager.initialize_clients()
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to update API key: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update API key: {str(e)}"
        ) 