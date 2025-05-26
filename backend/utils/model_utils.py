import os
import logging
from typing import Dict, Any, Optional
import dspy
from openai import OpenAI
from anthropic import Anthropic

# Optional imports
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    Mistral = None
    MISTRAL_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    cohere = None
    COHERE_AVAILABLE = False

import requests
from config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model configurations and connections for different AI providers."""
    
    def __init__(self):
        self.clients = {}
        self.initialize_clients()
        self.configure_dspy()
    
    def initialize_clients(self):
        """Initialize API clients for available providers."""
        # OpenAI client
        if settings.OPENAI_API_KEY:
            try:
                self.clients["openai"] = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Claude client
        if settings.ANTHROPIC_API_KEY:
            try:
                self.clients["claude"] = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("Claude client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {str(e)}")
        
        # Google client
        if settings.GOOGLE_API_KEY and GOOGLE_AVAILABLE:
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.clients["google"] = genai
                logger.info("Google Gemini client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google client: {str(e)}")
        
        # Mistral client
        if settings.MISTRAL_API_KEY and MISTRAL_AVAILABLE:
            try:
                self.clients["mistral"] = Mistral(api_key=settings.MISTRAL_API_KEY)
                logger.info("Mistral client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral client: {str(e)}")
        
        # Cohere client
        if settings.COHERE_API_KEY and COHERE_AVAILABLE:
            try:
                self.clients["cohere"] = cohere.Client(settings.COHERE_API_KEY)
                logger.info("Cohere client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client: {str(e)}")
        
        # Groq client
        if settings.GROQ_API_KEY:
            try:
                self.clients["groq"] = OpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                logger.info("Groq client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                
        # Ollama client (local)
        try:
            # Test the connection to Ollama
            response = requests.get(f"{settings.OLLAMA_API_BASE}/api/tags")
            if response.status_code == 200:
                self.clients["ollama"] = {"base_url": settings.OLLAMA_API_BASE}
                logger.info("Ollama client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
    
    def configure_dspy(self):
        """Configure DSPy with the appropriate LM backend."""
        provider = settings.DSPY_SETTINGS["provider"]
        model = settings.DSPY_SETTINGS["model"]
        
        try:
            # Configure DSPy based on the provider
            if provider == "openai" and "openai" in self.clients:
                api_key = settings.OPENAI_API_KEY
                if api_key:
                    lm = dspy.OpenAI(model=model, api_key=api_key)
                    dspy.configure(lm=lm)
                    logger.info(f"DSPy configured with OpenAI model: {model}")
                    return
                    
            elif provider == "claude" and "claude" in self.clients:
                api_key = settings.ANTHROPIC_API_KEY
                if api_key:
                    # Use a simple LM wrapper for Claude that works with DSPy
                    try:
                        # Create a custom LM class for Claude
                        class ClaudeLM:
                            def __init__(self, model, api_key):
                                self.model = model
                                self.api_key = api_key
                                self.client = Anthropic(api_key=api_key)
                                # Add required attributes for DSPy compatibility
                                self.kwargs = {
                                    "temperature": 0.0,
                                    "max_tokens": 1000,
                                    "top_p": 1.0
                                }
                                self.history = []
                            
                            def __call__(self, prompt, **kwargs):
                                try:
                                    # Merge default kwargs with provided ones
                                    merged_kwargs = {**self.kwargs, **kwargs}
                                    
                                    response = self.client.messages.create(
                                        model=self.model,
                                        max_tokens=merged_kwargs.get('max_tokens', 1000),
                                        temperature=merged_kwargs.get('temperature', 0.0),
                                        messages=[{"role": "user", "content": prompt}]
                                    )
                                    result = response.content[0].text
                                    
                                    # Store in history for DSPy compatibility
                                    self.history.append({
                                        "prompt": prompt,
                                        "response": result,
                                        "kwargs": merged_kwargs
                                    })
                                    
                                    return [result]
                                except Exception as e:
                                    logger.error(f"Claude API call failed: {str(e)}")
                                    return ["Error: Could not generate response"]
                        
                        lm = ClaudeLM(model, api_key)
                        dspy.configure(lm=lm)
                        logger.info(f"DSPy configured with Claude model: {model}")
                        return
                    except Exception as e:
                        logger.error(f"Failed to configure Claude for DSPy: {str(e)}")
                    
            elif provider == "groq" and "groq" in self.clients:
                api_key = settings.GROQ_API_KEY
                if api_key:
                    lm = dspy.OpenAI(
                        model=model,
                        api_key=api_key,
                        api_base="https://api.groq.com/openai/v1"
                    )
                    dspy.configure(lm=lm)
                    logger.info(f"DSPy configured with Groq model: {model}")
                    return
            
            # If no specific provider works, try to fallback to any available provider
            logger.warning("Primary provider not available, trying fallback options")
            
            # Try OpenAI as fallback
            if "openai" in self.clients and settings.OPENAI_API_KEY:
                lm = dspy.OpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
                dspy.configure(lm=lm)
                logger.info("DSPy configured with OpenAI fallback")
                return
            
            # If no providers work, use basic configuration
            logger.warning("No suitable provider found for DSPy, using basic configuration")
            dspy.configure()
            
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {str(e)}")
            # Fallback to basic configuration
            try:
                dspy.configure()
                logger.info("DSPy configured with basic fallback")
            except Exception as fallback_error:
                logger.error(f"DSPy fallback configuration failed: {str(fallback_error)}")
    
    def get_client(self, provider: str):
        """Get client for the specified provider."""
        if provider in self.clients:
            return self.clients[provider]
        return None
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available providers and models."""
        available = {}
        
        for provider, config in settings.AI_PROVIDERS.items():
            if provider in self.clients:
                available[provider] = {
                    "models": config["models"],
                    "default_model": config["default_model"]
                }
        
        return available
    
    def update_dspy_config(self, provider: str, model: str):
        """Update DSPy configuration with new provider and model."""
        # If provider not specified, use current provider
        if provider is None:
            provider = settings.DSPY_SETTINGS["provider"]
            logger.info(f"Using current provider: {provider}")
        
        # Update settings
        previous_provider = settings.DSPY_SETTINGS["provider"]
        previous_model = settings.DSPY_SETTINGS["model"]
        
        settings.DSPY_SETTINGS["provider"] = provider
        settings.DSPY_SETTINGS["model"] = model
        
        # Configure DSPy with the new settings
        try:
            self.configure_dspy()
            return True
        except Exception as e:
            # If configuration fails, revert to previous settings
            logger.error(f"Failed to update DSPy config: {str(e)}")
            settings.DSPY_SETTINGS["provider"] = previous_provider
            settings.DSPY_SETTINGS["model"] = previous_model
            self.configure_dspy()
            return False

# Create singleton instance
model_manager = ModelManager() 