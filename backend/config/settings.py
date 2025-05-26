import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Default model settings
DEFAULT_AI_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", "claude")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")

# Server settings
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# Available AI providers and models
AI_PROVIDERS = {
    "openai": {
        "api_key_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "default_model": "gpt-4o"
    },
    "claude": {
        "api_key_var": "ANTHROPIC_API_KEY",
        "models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default_model": "claude-3-7-sonnet-20250219"
    },
    "google": {
        "api_key_var": "GOOGLE_API_KEY",
        "models": ["gemini-2-5-flash", "gemini-2-5-pro"],
        "default_model": "gemini-2-5-flash"
    },
    "mistral": {
        "api_key_var": "MISTRAL_API_KEY",
        "models": ["mistral-large", "mistral-medium", "mistral-small"],
        "default_model": "mistral-small"
    },
    "cohere": {
        "api_key_var": "COHERE_API_KEY",
        "models": ["command", "command-light", "command-r", "command-r-plus"],
        "default_model": "command"
    },
    "groq": {
        "api_key_var": "GROQ_API_KEY",
        "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b"],
        "default_model": "llama3-8b-8192"
    },
    "ollama": {
        "api_base_var": "OLLAMA_API_BASE",
        "models": ["llama3", "llama2", "mistral", "phi3", "orca-mini"],
        "default_model": "llama3"
    }
}

# DSPy configuration - Use Claude with custom implementation
DSPY_SETTINGS = {
    "provider": "claude",  # Use Claude with custom DSPy wrapper
    "model": "claude-3-5-sonnet-20241022"  # Use the same model as default
} 