from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from api.chat_routes import router as chat_router
from api.analytics_routes import router as analytics_router

# Import managers
from managers.global_managers import ai_manager, session_manager

# Import utilities
from utils.logger import Logger

# Set up logging
logger = Logger("main", see_time=True, console_log=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    # Startup
    logger.log_message("Starting DSAgency Auto-Analyst Backend", level=logging.INFO)
    
    # Try to configure default AI model if environment variables are set
    try:
        # Try OpenAI first
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            result = ai_manager.configure_model("openai", "gpt-4", openai_key)
            if result["status"] == "success":
                logger.log_message("Default OpenAI model configured successfully", level=logging.INFO)
            else:
                logger.log_message(f"Failed to configure OpenAI model: {result.get('error')}", level=logging.WARNING)
        
        # Try Anthropic if OpenAI not available
        elif os.getenv("ANTHROPIC_API_KEY"):
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            # Try with the latest Claude Sonnet 4 model
            result = ai_manager.configure_model("anthropic", "claude-sonnet-4-20250514", anthropic_key)
            if result["status"] == "success":
                logger.log_message("Default Anthropic Claude Sonnet 4 model configured successfully", level=logging.INFO)
            else:
                # Fallback to Claude 3.5 Sonnet if Claude 4 fails
                logger.log_message(f"Claude 4 failed, trying Claude 3.5: {result.get('error')}", level=logging.WARNING)
                result = ai_manager.configure_model("anthropic", "claude-3-5-sonnet-20241022", anthropic_key)
                if result["status"] == "success":
                    logger.log_message("Default Anthropic Claude 3.5 Sonnet model configured successfully", level=logging.INFO)
                else:
                    logger.log_message(f"Failed to configure Anthropic model: {result.get('error')}", level=logging.WARNING)
        
        else:
            logger.log_message("No AI API keys found in environment variables", level=logging.WARNING)
            
    except Exception as e:
        logger.log_message(f"Error configuring default model: {str(e)}", level=logging.WARNING)
    
    yield
    
    # Shutdown
    logger.log_message("Shutting down DSAgency Auto-Analyst Backend", level=logging.INFO)
    
    # Cleanup inactive sessions
    try:
        cleaned = session_manager.cleanup_inactive_sessions(max_age_hours=24)
        logger.log_message(f"Cleaned up {cleaned} inactive sessions", level=logging.INFO)
    except Exception as e:
        logger.log_message(f"Error during session cleanup: {str(e)}", level=logging.WARNING)

# Create FastAPI app
app = FastAPI(
    title="DSAgency Auto-Analyst",
    description="AI-powered data analysis and chat system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(analytics_router)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with system information.
    """
    return {
        "message": "DSAgency Auto-Analyst Backend",
        "version": "1.0.0",
        "status": "running",
        "ai_configured": ai_manager.is_configured(),
        "endpoints": {
            "chat": "/api/chat",
            "analytics": "/api/analytics",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "ai_configured": ai_manager.is_configured(),
        "ai_config": ai_manager.get_current_config(),
        "active_sessions": len(session_manager.active_sessions)
    }

# AI configuration endpoints
@app.post("/api/configure-ai")
async def configure_ai(provider: str, model: str, api_key: str = None):
    """
    Configure AI model for the system.
    """
    try:
        result = ai_manager.configure_model(provider, model, api_key)
        
        if result["status"] == "success":
            logger.log_message(f"AI model configured: {provider}/{model}", level=logging.INFO)
            return result
        else:
            logger.log_message(f"Failed to configure AI model: {result.get('error')}", level=logging.ERROR)
            raise HTTPException(status_code=400, detail=result.get("error", "Configuration failed"))
            
    except Exception as e:
        logger.log_message(f"Error configuring AI: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai-config")
async def get_ai_config():
    """
    Get current AI configuration.
    """
    try:
        config = ai_manager.get_current_config()
        return {
            "configured": ai_manager.is_configured(),
            "config": config
        }
    except Exception as e:
        logger.log_message(f"Error getting AI config: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-models")
async def get_available_models():
    """
    Get list of available models for each provider.
    """
    try:
        models = ai_manager.get_available_models()
        return {
            "status": "success",
            "models": models
        }
    except Exception as e:
        logger.log_message(f"Error getting available models: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.log_message(f"Unhandled exception: {str(exc)}", level=logging.ERROR)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.log_message(f"Starting server on {host}:{port}", level=logging.INFO)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )
