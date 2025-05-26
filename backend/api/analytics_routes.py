from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
import io
import json
import logging
from datetime import datetime

# Import managers
from managers.session_manager import SessionManager
from managers.ai_manager import AIManager

# Import agents
from agents.agents import auto_analyst, auto_analyst_ind, dataset_description_agent, chat_history_name_agent

# Import retrievers
from agents.retrievers.retrievers import make_data, styling_instructions

# Import utilities
from utils.logger import Logger

# Set up logging
logger = Logger("analytics_routes", see_time=True, console_log=False)

# Create router
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Initialize managers
session_manager = SessionManager()
ai_manager = AIManager()

# Global variables for dataset and agents
current_dataset = None
dataset_description = None
retrievers = None
analyst_agent = None
individual_agent = None

class DatasetUploadResponse(BaseModel):
    message: str
    dataset_info: Dict[str, Any]
    columns: List[str]
    shape: List[int]

class AnalysisRequest(BaseModel):
    query: str
    agent: Optional[str] = None
    session_id: Optional[str] = None

@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a dataset for analysis.
    """
    try:
        # Check if AI is configured
        if not ai_manager.is_configured():
            raise HTTPException(
                status_code=400, 
                detail="AI model not configured. Please configure a model first."
            )
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read the file
        contents = await file.read()
        
        # Load dataset based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Store dataset globally
        global current_dataset, dataset_description, retrievers, analyst_agent, individual_agent
        current_dataset = df
        
        # Generate dataset description if not provided
        if not description:
            description = f"Dataset uploaded from {file.filename}"
        
        # Create dataset metadata
        dataset_info = make_data(df, description)
        
        # Generate enhanced description using AI
        try:
            from agents.agents import dataset_description_agent
            import dspy
            
            desc_agent = dspy.ChainOfThought(dataset_description_agent)
            enhanced_desc = desc_agent(
                dataset=str(dataset_info),
                existing_description=description
            )
            dataset_description = enhanced_desc.description
        except Exception as e:
            logger.log_message(f"Error generating enhanced description: {str(e)}", level=logging.WARNING)
            dataset_description = description
        
        # Create retrievers
        class SimpleRetriever:
            def __init__(self, content):
                self.content = content
            
            def as_retriever(self):
                return self
            
            def retrieve(self, query):
                class Result:
                    def __init__(self, text):
                        self.text = text
                return [Result(self.content)]
        
        retrievers = {
            "dataframe_index": SimpleRetriever(str(dataset_info)),
            "style_index": SimpleRetriever("\n".join(styling_instructions))
        }
        
        # Initialize agents with the new dataset
        analyst_agent = auto_analyst({}, retrievers)
        individual_agent = auto_analyst_ind({}, retrievers)
        
        logger.log_message(f"Dataset uploaded successfully: {file.filename}, Shape: {df.shape}", level=logging.INFO)
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset_info": {
                "filename": file.filename,
                "description": dataset_description,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records')
            },
            "columns": list(df.columns),
            "shape": list(df.shape)
        }
        
    except Exception as e:
        logger.log_message(f"Error uploading dataset: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset-info")
async def get_dataset_info():
    """
    Get information about the currently loaded dataset.
    """
    try:
        global current_dataset, dataset_description
        
        if current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")
        
        return {
            "description": dataset_description,
            "shape": list(current_dataset.shape),
            "columns": list(current_dataset.columns),
            "dtypes": current_dataset.dtypes.astype(str).to_dict(),
            "null_counts": current_dataset.isnull().sum().to_dict(),
            "sample_data": current_dataset.head(5).to_dict('records'),
            "summary_stats": current_dataset.describe().to_dict() if len(current_dataset.select_dtypes(include='number').columns) > 0 else {}
        }
        
    except Exception as e:
        logger.log_message(f"Error getting dataset info: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """
    Perform data analysis using the auto-analyst system.
    """
    try:
        # Check if dataset is loaded
        if current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a dataset first.")
        
        # Check if AI is configured
        if not ai_manager.is_configured():
            raise HTTPException(
                status_code=400, 
                detail="AI model not configured. Please configure a model first."
            )
        
        # Create or get session
        if request.session_id:
            session = session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = request.session_id
        else:
            session_id = session_manager.create_session()
        
        # Add user message to session
        user_message = {
            "role": "user",
            "content": request.query,
            "timestamp": datetime.utcnow().isoformat()
        }
        session_manager.add_message(session_id, user_message)
        
        # Process the analysis request
        global analyst_agent, individual_agent
        
        if request.agent:
            # Direct agent routing
            result = individual_agent.forward(request.query, request.agent)
            agent_used = request.agent
        else:
            # Use planner-based routing
            result = analyst_agent.forward(request.query)
            agent_used = "auto_analyst"
        
        # Format response
        if isinstance(result, dict):
            if "error" in result:
                response_content = f"Error: {result['error']}"
                analysis_result = {"error": result['error']}
            else:
                response_content = str(result)
                analysis_result = result
        else:
            response_content = str(result)
            analysis_result = {"result": str(result)}
        
        # Add assistant message to session
        assistant_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "agent_used": agent_used,
                "analysis_type": "data_analysis"
            }
        }
        session_manager.add_message(session_id, assistant_message)
        
        logger.log_message(f"Analysis completed - Session: {session_id}, Agent: {agent_used}", level=logging.INFO)
        
        return {
            "response": response_content,
            "analysis_result": analysis_result,
            "session_id": session_id,
            "agent_used": agent_used
        }
        
    except Exception as e:
        logger.log_message(f"Error in analysis: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_analytics_agents():
    """
    List available analytics agents.
    """
    try:
        from agents.agents import AGENTS_WITH_DESCRIPTION, PLANNER_AGENTS_WITH_DESCRIPTION
        
        agents = {}
        
        # Add individual agents
        for agent_name, description in AGENTS_WITH_DESCRIPTION.items():
            agents[agent_name] = {
                "name": agent_name,
                "description": description,
                "type": "individual",
                "category": "analytics"
            }
        
        # Add planner agents
        for agent_name, description in PLANNER_AGENTS_WITH_DESCRIPTION.items():
            agents[agent_name] = {
                "name": agent_name,
                "description": description,
                "type": "planner",
                "category": "analytics"
            }
        
        return {"agents": agents}
        
    except Exception as e:
        logger.log_message(f"Error listing analytics agents: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-chat-title")
async def generate_chat_title(query: str):
    """
    Generate a title for a chat based on the query.
    """
    try:
        # Check if AI is configured
        if not ai_manager.is_configured():
            raise HTTPException(
                status_code=400, 
                detail="AI model not configured. Please configure a model first."
            )
        
        # Use the chat history name agent
        import dspy
        name_agent = dspy.ChainOfThought(chat_history_name_agent)
        result = name_agent(query=query)
        
        return {"title": result.name}
        
    except Exception as e:
        logger.log_message(f"Error generating chat title: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/dataset")
async def clear_dataset():
    """
    Clear the currently loaded dataset.
    """
    try:
        global current_dataset, dataset_description, retrievers, analyst_agent, individual_agent
        
        current_dataset = None
        dataset_description = None
        retrievers = None
        analyst_agent = None
        individual_agent = None
        
        logger.log_message("Dataset cleared successfully", level=logging.INFO)
        
        return {"message": "Dataset cleared successfully"}
        
    except Exception as e:
        logger.log_message(f"Error clearing dataset: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e)) 