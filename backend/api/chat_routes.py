from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
import json
import time
from datetime import datetime
import os
import aiofiles
import uuid
import pandas as pd
import numpy as np
from io import StringIO
import re
import httpx

# Import managers
from managers.session_manager import SessionManager
from managers.ai_manager import AIManager

# Import schemas
from schemas.query_schemas import QueryRequest, QueryResponse, ChatMessage, SessionRequest, SessionResponse

# Import agents
from agents.agents import auto_analyst, auto_analyst_ind, get_agent_description, get_multi_agent_system
from agents.web_search_agent import get_web_search_agent
from agents.code_execution_agent import execute_and_analyze_code, get_code_executor

# Import utilities
from utils.logger import Logger
from utils.code_formatter import format_python_code, create_executable_code_block

# Import global instances from the global managers module
from managers.global_managers import ai_manager, session_manager

# Set up logging
logger = Logger("chat_routes", see_time=True, console_log=False)

# Create router
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize agents (will be done when AI is configured)
analyst_agent = None
individual_agent = None

# Initialize AI Manager
ai_manager = AIManager()

def analyze_csv_file(file_path: str) -> dict:
    """
    Analyze a CSV file and return comprehensive EDA information.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic information
        basic_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        # Statistical summary
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        statistical_summary = {}
        if numeric_columns:
            desc = df[numeric_columns].describe()
            statistical_summary["numeric"] = {}
            for col in desc.columns:
                statistical_summary["numeric"][col] = {
                    "count": float(desc.loc['count', col]),
                    "mean": float(desc.loc['mean', col]),
                    "std": float(desc.loc['std', col]),
                    "min": float(desc.loc['min', col]),
                    "25%": float(desc.loc['25%', col]),
                    "50%": float(desc.loc['50%', col]),
                    "75%": float(desc.loc['75%', col]),
                    "max": float(desc.loc['max', col])
                }
        
        if categorical_columns:
            statistical_summary["categorical"] = {}
            for col in categorical_columns:
                statistical_summary["categorical"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in df[col].value_counts().head(5).items()},
                    "missing_count": int(df[col].isnull().sum())
                }
        
        # Missing values analysis
        missing_values = {
            "total_missing": {col: int(count) for col, count in df.isnull().sum().items()},
            "missing_percentage": {col: float(pct) for col, pct in (df.isnull().sum() / len(df) * 100).items()}
        }
        
        # Correlation analysis (only for numeric columns)
        correlation_matrix = {}
        if len(numeric_columns) > 1:
            corr = df[numeric_columns].corr()
            correlation_matrix = {
                col1: {col2: float(corr.loc[col1, col2]) if not pd.isna(corr.loc[col1, col2]) else None 
                       for col2 in corr.columns}
                for col1 in corr.index
            }
        
        # Sample data - convert to basic Python types
        sample_data = []
        for _, row in df.head(10).iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    row_dict[col] = float(val)
                else:
                    row_dict[col] = str(val)
            sample_data.append(row_dict)
        
        return {
            "basic_info": basic_info,
            "statistical_summary": statistical_summary,
            "missing_values": missing_values,
            "correlation_matrix": correlation_matrix,
            "sample_data": sample_data,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns
        }
        
    except Exception as e:
        return {"error": f"Error analyzing CSV file: {str(e)}"}

def get_uploaded_files_info() -> list:
    """
    Get information about all uploaded files.
    """
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            return []
        
        files_info = []
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                file_extension = os.path.splitext(filename)[1].lower()
                
                file_info = {
                    "filename": filename,
                    "size": stat.st_size,
                    "extension": file_extension,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "path": file_path
                }
                
                # If it's a CSV file, add basic analysis
                if file_extension == '.csv':
                    try:
                        df = pd.read_csv(file_path)
                        file_info["csv_info"] = {
                            "rows": len(df),
                            "columns": len(df.columns),
                            "column_names": list(df.columns)
                        }
                    except:
                        file_info["csv_info"] = {"error": "Could not read CSV"}
                
                files_info.append(file_info)
        
        return files_info
    except Exception as e:
        return []

class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    file_path: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

class SimpleChatRequest(BaseModel):
    message: str
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

class AgentRequest(BaseModel):
    agent_name: str
    query: str
    session_id: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

def format_ai_response(response_text: str) -> str:
    """
    Format AI response to improve code blocks and overall presentation.
    
    Args:
        response_text: Raw response from AI
        
    Returns:
        Formatted response with improved code blocks
    """
    try:
        # Temporarily disable formatting to debug the issue
        return response_text
        
        # Find all Python code blocks in the response
        code_block_pattern = r'```python\n(.*?)\n```'
        
        def format_code_block(match):
            code_content = match.group(1)
            # Format the code using our formatter
            formatted_code = format_python_code(code_content)
            return f'```python\n{formatted_code}\n```'
        
        # Replace all Python code blocks with formatted versions
        formatted_response = re.sub(code_block_pattern, format_code_block, response_text, flags=re.DOTALL)
        
        # Also handle cases where code might not be in proper markdown blocks
        # Look for lines that seem like Python code without proper formatting
        lines = formatted_response.split('\n')
        in_code_block = False
        formatted_lines = []
        potential_code_lines = []
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                formatted_lines.append(line)
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                formatted_lines.append(line)
            elif in_code_block:
                formatted_lines.append(line)
            else:
                # Check if this line looks like Python code that should be in a block
                if _looks_like_python_code(line) and not in_code_block:
                    potential_code_lines.append(line)
                else:
                    # If we had potential code lines, format them
                    if potential_code_lines:
                        code_content = '\n'.join(potential_code_lines)
                        formatted_code = format_python_code(code_content)
                        formatted_lines.append(f'```python\n{formatted_code}\n```')
                        potential_code_lines = []
                    formatted_lines.append(line)
        
        # Handle any remaining potential code lines
        if potential_code_lines:
            code_content = '\n'.join(potential_code_lines)
            formatted_code = format_python_code(code_content)
            formatted_lines.append(f'```python\n{formatted_code}\n```')
        
        return '\n'.join(formatted_lines)
        
    except Exception as e:
        logger.log_message(f"Error formatting AI response: {str(e)}", level=logging.WARNING)
        return response_text  # Return original if formatting fails

def _looks_like_python_code(line: str) -> bool:
    """
    Check if a line looks like Python code that should be in a code block.
    
    Args:
        line: Line of text to check
        
    Returns:
        True if line looks like Python code
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Common Python patterns
    python_patterns = [
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'^plt\.',
        r'^np\.',
        r'^pd\.',
        r'^sns\.',
        r'^\w+\s*=\s*.+',
        r'^print\s*\(',
        r'^def\s+\w+\s*\(',
        r'^class\s+\w+',
        r'^if\s+.+:',
        r'^for\s+.+:',
        r'^while\s+.+:',
        r'^try\s*:',
        r'^except\s*.*:',
        r'^with\s+.+:',
    ]
    
    return any(re.match(pattern, stripped) for pattern in python_patterns)

def is_simple_conversational_query(message: str) -> bool:
    """
    Determine if a message is a simple conversational query that doesn't need code execution.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if it's a simple conversational query
    """
    message_lower = message.lower().strip()
    
    # Simple greeting patterns
    greeting_patterns = [
        r'^(hi|hello|hola|hey|good morning|good afternoon|good evening)',
        r'^(how are you|como estas|what\'s up|que tal)',
        r'^(my name is|me llamo|i am|soy)',
        r'^(what is|que es|what are|que son)',
        r'^(can you|puedes|could you|podrias)',
        r'^(tell me about|cuentame sobre|explain|explica)',
        r'^(what agents|que agentes|what can you do|que puedes hacer)',
        r'^(help|ayuda|how do i|como hago)',
    ]
    
    # Check if it matches simple conversational patterns
    for pattern in greeting_patterns:
        if re.match(pattern, message_lower):
            return True
    
    # Check if it's a simple question without data analysis intent
    simple_question_keywords = [
        "what is", "que es", "what are", "que son", "how do i", "como hago",
        "can you explain", "puedes explicar", "tell me", "cuentame",
        "what agents", "que agentes", "available", "disponible"
    ]
    
    # If it's a short message with simple question keywords, it's probably conversational
    if len(message.split()) < 15 and any(keyword in message_lower for keyword in simple_question_keywords):
        return True
    
    # Check if it explicitly mentions data analysis or code
    data_analysis_keywords = [
        "dataset", "data", "csv", "excel", "analyze", "analiza", "visualization", 
        "visualización", "plot", "graph", "modelo", "model", "prediction", "predicción"
    ]
    
    if any(keyword in message_lower for keyword in data_analysis_keywords):
        return False
    
    # If none of the above, consider it conversational if it's short
    return len(message.split()) < 10

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user messages and returns AI responses.
    Uses the multi-agent system for data analysis requests and direct AI for conversations.
    """
    try:
        logger.log_message(f"Chat request received: {request.message[:100]}...", level=logging.INFO)
        
        # Configure model if specified
        if request.model_provider and request.model_name:
            logger.log_message(f"Configuring model: {request.model_provider}/{request.model_name}", level=logging.INFO)
            config_result = ai_manager.configure_model(
                provider=request.model_provider,
                model=request.model_name
            )
            if config_result["status"] != "success":
                logger.log_message(f"Model configuration failed: {config_result['message']}", level=logging.ERROR)
                raise HTTPException(status_code=400, detail=config_result["message"])
        
        # Check if AI manager is configured
        logger.log_message(f"Checking if AI manager is configured: {ai_manager.is_configured()}", level=logging.INFO)
        if not ai_manager.is_configured():
            logger.log_message("AI manager not configured, attempting default configuration", level=logging.WARNING)
            # Try to configure with default Anthropic model
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                config_result = ai_manager.configure_model(
                    provider='anthropic',
                    model='claude-3-5-sonnet-20241022',
                    api_key=anthropic_key
                )
                logger.log_message(f"Default configuration result: {config_result}", level=logging.INFO)
                if config_result["status"] != "success":
                    raise HTTPException(
                        status_code=500, 
                        detail="AI model not configured. Please configure a model first."
                    )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="AI model not configured. Please configure a model first."
                )
        
        # Get or create session for conversation history
        from managers.global_managers import session_manager
        
        session_id = request.session_id
        session = session_manager.get_session(session_id)
        
        if not session:
            # Create new session
            session_manager.sessions[session_id] = {
                "session_id": session_id,
                "user_id": None,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "messages": [],
                "context": {},
                "metadata": {}
            }
            session_manager.active_sessions.add(session_id)
            logger.log_message(f"Created new session: {session_id}", level=logging.INFO)
        
        # Add user message to session
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        }
        session_manager.add_message(session_id, user_message)
        
        # Get conversation history for context
        session = session_manager.get_session(session_id)
        messages = []
        
        # Convert session messages to AI format (keep conversation history)
        if session and "messages" in session:
            for msg in session["messages"]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Ensure we have at least the current message
        if not messages:
            messages = [{
                "role": "user",
                "content": request.message
            }]
        
        logger.log_message(f"Prepared messages: {len(messages)} messages", level=logging.INFO)
        
        # Add file context if provided
        file_context = ""
        if request.file_path:
            try:
                # Read file content (implement file reading logic here)
                file_context = f"User has uploaded a file: {request.file_path}"
                messages[-1]["content"] = f"{file_context}\n\nUser question: {request.message}"
                logger.log_message(f"Added file context for: {request.file_path}", level=logging.INFO)
            except Exception as e:
                logger.log_message(f"Error reading file {request.file_path}: {str(e)}", level=logging.WARNING)
        
        # Determine if this should use the multi-agent system or direct AI
        # Use direct AI by default for better personality and conversation flow
        
        start_time = time.time()
        
        # Keywords that indicate complex analysis requiring multi-agent system
        complex_analysis_keywords = [
            "eda", "exploratory data analysis", "análisis exploratorio",
            "machine learning model", "regression analysis", "classification model", 
            "clustering analysis", "deep learning", "neural network", "complex workflow",
            "build a model", "train a model", "create a pipeline", "data analysis",
            "analyze", "analiza", "analizar", "dataset analysis", "statistical analysis",
            "correlation analysis", "feature engineering", "data preprocessing",
            "missing values analysis", "outlier detection", "data visualization",
            "distribution analysis", "hypothesis testing"
        ]
        
        # Keywords that indicate the user wants code execution
        code_execution_keywords = [
            "execute", "run", "ejecuta", "corre", "analyze this data", "analiza estos datos",
            "load the data", "carga los datos", "show me the results", "muestra los resultados",
            "create visualization", "crea visualización", "plot", "gráfico", "execute this code",
            "run this code", "ejecuta este código", "corre este código", "show me", "muéstrame"
        ]
        
        is_complex_analysis = any(keyword in request.message.lower() for keyword in complex_analysis_keywords)
        wants_code_execution = any(keyword in request.message.lower() for keyword in code_execution_keywords)
        
        # If user uploaded a file, they probably want analysis - use multi-agent system
        has_uploaded_file = bool(request.file_path)
        
        # Use multi-agent system if:
        # 1. Complex analysis keywords detected
        # 2. File uploaded (likely wants analysis)
        # 3. EDA or data analysis mentioned
        # Using more flexible logic - file uploaded AND analysis requested
        should_use_multi_agent = (
            has_uploaded_file and 
            (is_complex_analysis or 
             any(keyword in request.message.lower() for keyword in ["eda", "exploratory", "análisis exploratorio", "exploratory data analysis", "análisis de datos"]))
        )
        
        # Debug logging
        logger.log_message(f"Multi-agent detection - has_file: {has_uploaded_file}, is_complex: {is_complex_analysis}, should_use: {should_use_multi_agent}", level=logging.INFO)
        
        if should_use_multi_agent:
            logger.log_message("Using multi-agent system for data analysis request", level=logging.INFO)
            
            # Use multi-agent system for data analysis
            multi_agent_system = get_multi_agent_system()
            
            # Prepare context with file information
            analysis_context = ""
            if request.file_path:
                try:
                    # Get file info and add to context
                    file_name = os.path.basename(request.file_path)
                    file_size = os.path.getsize(request.file_path) if os.path.exists(request.file_path) else 0
                    
                    # If it's a CSV file, get detailed analysis
                    if file_name.lower().endswith('.csv'):
                        csv_analysis = analyze_csv_file(request.file_path)
                        if 'error' not in csv_analysis:
                            analysis_context = f"""User uploaded CSV file: {file_name} (size: {file_size} bytes)
File path: {request.file_path}

Dataset Information:
- Shape: {csv_analysis['basic_info']['shape'][0]} rows, {csv_analysis['basic_info']['shape'][1]} columns
- Columns: {', '.join(csv_analysis['basic_info']['columns'])}
- Memory usage: {csv_analysis['basic_info']['memory_usage']} bytes

Data Types:
{chr(10).join([f"- {col}: {dtype}" for col, dtype in csv_analysis['basic_info']['data_types'].items()])}

Numeric columns: {', '.join(csv_analysis['numeric_columns']) if csv_analysis['numeric_columns'] else 'None'}
Categorical columns: {', '.join(csv_analysis['categorical_columns']) if csv_analysis['categorical_columns'] else 'None'}

Missing Values:
{chr(10).join([f"- {col}: {count} missing ({pct:.1f}%)" for col, count, pct in zip(csv_analysis['missing_values']['total_missing'].keys(), csv_analysis['missing_values']['total_missing'].values(), csv_analysis['missing_values']['missing_percentage'].values()) if count > 0]) if any(csv_analysis['missing_values']['total_missing'].values()) else '- No missing values detected'}

Sample Data (first 3 rows):
{chr(10).join([str(row) for row in csv_analysis['sample_data'][:3]])}"""
                        else:
                            analysis_context = f"User uploaded file: {file_name} (size: {file_size} bytes)\nFile path: {request.file_path}\nError analyzing CSV: {csv_analysis['error']}"
                    else:
                        analysis_context = f"User uploaded file: {file_name} (size: {file_size} bytes)\nFile path: {request.file_path}"
                    
                    logger.log_message(f"Added detailed file context: {file_name}", level=logging.INFO)
                except Exception as e:
                    logger.log_message(f"Error getting file info: {str(e)}", level=logging.WARNING)
                    analysis_context = f"User uploaded file: {request.file_path}"
            
            # Route the query through the multi-agent system
            routing_decision = multi_agent_system.route_query(request.message, analysis_context)
            logger.log_message(f"Multi-agent routing decision: {routing_decision}", level=logging.INFO)
            
            # Execute the workflow
            workflow_result = multi_agent_system.execute_workflow(
                user_query=request.message,
                available_data=analysis_context or "No specific dataset provided"
            )
            
            logger.log_message(f"Multi-agent workflow result type: {workflow_result.get('type', 'unknown')}", level=logging.INFO)
            logger.log_message(f"Multi-agent workflow result keys: {list(workflow_result.keys())}", level=logging.INFO)
            logger.log_message(f"Multi-agent workflow response length: {len(workflow_result.get('response', ''))}", level=logging.INFO)
            
            # Debug: Log the full response
            if 'response' in workflow_result:
                logger.log_message(f"Multi-agent response preview: {workflow_result['response'][:500]}...", level=logging.INFO)
            
            # Format multi-agent response
            if workflow_result["type"] == "analysis":
                # Use the response from the multi-agent system if available
                if "response" in workflow_result and workflow_result["response"]:
                    ai_response = workflow_result["response"]
                    logger.log_message(f"Using multi-agent response, length: {len(ai_response)}", level=logging.INFO)
                else:
                    logger.log_message("No response in workflow_result, creating from results", level=logging.WARNING)
                    # Create a comprehensive response from the results
                    response_parts = []
                    
                    # Add plan summary if available
                    if "plan" in workflow_result and hasattr(workflow_result["plan"], 'plan'):
                        response_parts.append(f"## 📋 Analysis Plan\n{workflow_result['plan'].plan}")
                    
                    # Add agent results
                    if "results" in workflow_result:
                        for agent_name, result in workflow_result["results"].items():
                            if isinstance(result, dict):
                                agent_title = agent_name.replace('_', ' ').replace('planner ', '').title()
                                
                                if "summary" in result:
                                    response_parts.append(f"## 🔍 {agent_title}\n{result['summary']}")
                                
                                if "code" in result and result["code"]:
                                    response_parts.append(f"### Generated Code:\n```python\n{result['code']}\n```")
                                
                                if "error" in result:
                                    response_parts.append(f"### ❌ Error in {agent_title}:\n{result['error']}")
                    
                    # Add execution summary
                    if "integration_summary" in workflow_result:
                        response_parts.append(f"## ✅ Execution Summary\n{workflow_result['integration_summary']}")
                    
                    ai_response = "\n\n".join(response_parts) if response_parts else "Multi-agent analysis completed."
                    logger.log_message(f"Created response from parts, length: {len(ai_response)}", level=logging.INFO)
            elif workflow_result["type"] == "conversational":
                ai_response = workflow_result["response"]
            elif workflow_result["type"] == "error":
                ai_response = f"❌ **Error in Multi-Agent System**: {workflow_result.get('message', 'Unknown error occurred')}"
            else:
                ai_response = workflow_result.get("message", "I encountered an issue processing your request.")
                
        else:
            logger.log_message("Using direct AI for request (conversational, simple analysis, code generation)", level=logging.INFO)
            
            # Enhance messages with file context if available
            enhanced_messages = messages.copy()
            
            # Add file context to the user message if file_path is provided
            if request.file_path:
                try:
                    file_name = os.path.basename(request.file_path)
                    file_size = os.path.getsize(request.file_path) if os.path.exists(request.file_path) else 0
                    
                    # Try to get CSV info if it's a CSV file
                    file_info = f"User uploaded file: {file_name} (size: {file_size} bytes)"
                    if file_name.lower().endswith('.csv'):
                        try:
                            df = pd.read_csv(request.file_path)
                            columns_info = ", ".join(df.columns.tolist())
                            file_info += f"\nCSV file with {len(df)} rows and {len(df.columns)} columns"
                            file_info += f"\nColumns: {columns_info}"
                            file_info += f"\nFile path for analysis: {request.file_path}"
                        except Exception as e:
                            logger.log_message(f"Error reading CSV file: {str(e)}", level=logging.WARNING)
                    
                    # Enhance the last user message with file context
                    if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                        enhanced_messages[-1]["content"] = f"{file_info}\n\nUser request: {enhanced_messages[-1]['content']}"
                    
                    logger.log_message(f"Enhanced message with file context: {file_name}", level=logging.INFO)
                    
                except Exception as e:
                    logger.log_message(f"Error adding file context: {str(e)}", level=logging.WARNING)
            
            # Use direct AI response for most requests (conversational, simple analysis, code generation)
            ai_response = ai_manager.generate_response(enhanced_messages)
        
        processing_time = time.time() - start_time
        logger.log_message(f"Response generated in {processing_time:.2f}s", level=logging.INFO)
        
        # Format the response to improve code blocks
        logger.log_message("Formatting AI response...", level=logging.INFO)
        formatted_response = format_ai_response(ai_response)
        
        # Only execute code if explicitly requested or for complex analysis
        final_response = formatted_response
        if (wants_code_execution or is_complex_analysis) and not is_simple_conversational_query(request.message):
            logger.log_message("User wants code execution - checking for Python code blocks...", level=logging.INFO)
            code_blocks = extract_python_code_blocks(formatted_response)
            
            if code_blocks:
                logger.log_message(f"Found {len(code_blocks)} Python code blocks, executing and analyzing...", level=logging.INFO)
                
                # Prepare data context for code execution
                data_context = f"User query: {request.message}"
                if request.file_path:
                    data_context += f"\nFile being analyzed: {request.file_path}"
                
                # Enhance response with code execution results
                final_response = enhance_response_with_code_execution(formatted_response, data_context)
                logger.log_message("Code execution and analysis completed", level=logging.INFO)
        else:
            logger.log_message("Skipping code execution - conversational query or no explicit request", level=logging.INFO)
        
        # Add assistant message to session
        assistant_message = {
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "processing_time": processing_time,
                "model_used": f"{ai_manager.current_provider}/{ai_manager.current_model}",
                "routing_decision": "complex_analysis" if is_complex_analysis else "direct_ai",
                "used_multi_agent": is_complex_analysis
            }
        }
        session_manager.add_message(session_id, assistant_message)
        
        # Get current model info
        current_config = ai_manager.get_current_config()
        model_used = f"{current_config.get('provider', 'unknown')}/{current_config.get('model', 'unknown')}"
        
        logger.log_message(f"Chat completed successfully with model: {model_used}", level=logging.INFO)
        
        return ChatResponse(
            response=final_response,
            session_id=request.session_id,
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error in chat endpoint: {str(e)}", level=logging.ERROR)
        import traceback
        logger.log_message(f"Traceback: {traceback.format_exc()}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/chat/agent")
async def agent_endpoint(request: AgentRequest):
    """
    Direct agent execution endpoint.
    """
    try:
        start_time = time.time()
        
        # Check if AI is configured
        if not ai_manager.is_configured():
            raise HTTPException(
                status_code=400, 
                detail="AI model not configured. Please configure a model first."
            )
        
        # Initialize individual agent if not done
        global individual_agent
        if individual_agent is None:
            retrievers = {"dataframe_index": None, "style_index": None}
            individual_agent = auto_analyst_ind({}, retrievers)
        
        # Execute the specific agent
        result = individual_agent.forward(request.query, request.agent_name)
        
        # Format response
        if isinstance(result, dict):
            if "error" in result:
                response_content = f"Error: {result['error']}"
            else:
                response_content = str(result)
        else:
            response_content = str(result)
        
        logger.log_message(f"Agent executed - Agent: {request.agent_name}", level=logging.INFO)
        
        return {
            "response": response_content,
            "agent_used": request.agent_name,
            "execution_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.log_message(f"Error in agent endpoint: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """
    List all sessions, optionally filtered by user.
    """
    try:
        sessions = session_manager.list_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.log_message(f"Error listing sessions: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/sessions")
async def create_session(request: SessionRequest):
    """
    Create a new chat session.
    """
    try:
        session_id = session_manager.create_session(request.user_id)
        session = session_manager.get_session(session_id)
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "message_count": 0
        }
    except Exception as e:
        logger.log_message(f"Error creating session: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session details and messages.
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "messages": session["messages"],
            "message_count": len(session["messages"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error getting session: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session.
    """
    try:
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error deleting session: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/agents")
async def list_agents():
    """
    List available agents and their descriptions.
    """
    try:
        from agents.agents import AGENTS_WITH_DESCRIPTION, PLANNER_AGENTS_WITH_DESCRIPTION
        
        agents = {}
        
        # Add individual agents
        for agent_name, description in AGENTS_WITH_DESCRIPTION.items():
            agents[agent_name] = {
                "name": agent_name,
                "description": description,
                "type": "individual"
            }
        
        # Add planner agents
        for agent_name, description in PLANNER_AGENTS_WITH_DESCRIPTION.items():
            agents[agent_name] = {
                "name": agent_name,
                "description": description,
                "type": "planner"
            }
        
        return {"agents": agents}
    except Exception as e:
        logger.log_message(f"Error listing agents: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/debug")
async def debug_chat():
    """
    Debug endpoint to check chat system status.
    """
    return {
        "status": "Chat system is running",
        "ai_configured": ai_manager.is_configured(),
        "ai_config": ai_manager.get_current_config(),
        "active_sessions": len(session_manager.active_sessions)
    }

@router.post("/chat/simple")
async def simple_chat(request: SimpleChatRequest):
    """
    Simplified chat endpoint without session management.
    """
    try:
        # Configure model if specified
        if request.model_provider and request.model_name:
            logger.log_message(f"Configuring model: {request.model_provider}/{request.model_name}", level=logging.INFO)
            config_result = ai_manager.configure_model(
                provider=request.model_provider,
                model=request.model_name
            )
            if config_result["status"] != "success":
                logger.log_message(f"Model configuration failed: {config_result['message']}", level=logging.ERROR)
                return {"error": config_result["message"]}
        
        if not ai_manager.is_configured():
            # Try to configure with default Anthropic model
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                config_result = ai_manager.configure_model(
                    provider='anthropic',
                    model='claude-3-5-sonnet-20241022',
                    api_key=anthropic_key
                )
                if config_result["status"] != "success":
                    return {"error": "AI model not configured. Please configure a model first."}
            else:
                return {"error": "AI model not configured. Please configure a model first."}
        
        # Simple message format
        messages = [{"role": "user", "content": request.message}]
        response = ai_manager.generate_response(messages)
        
        return {
            "response": response,
            "model": ai_manager.current_model,
            "provider": ai_manager.current_provider
        }
    except Exception as e:
        logger.log_message(f"Error in simple chat: {str(e)}", level=logging.ERROR)
        return {"error": str(e)}

@router.post("/chat/test")
async def test_chat():
    """
    Simple test endpoint to verify AI functionality.
    """
    try:
        if not ai_manager.is_configured():
            return {"error": "AI not configured"}
        
        messages = [{"role": "user", "content": "Hola, di solo 'Funciona correctamente'"}]
        response = ai_manager.generate_response(messages)
        
        return {
            "status": "success",
            "response": response,
            "model": ai_manager.current_model,
            "provider": ai_manager.current_provider
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/welcome")
async def welcome():
    """
    Welcome endpoint with system information.
    """
    return {
        "message": "Bienvenido a DSAgency Auto-Analyst",
        "version": "1.0.0",
        "ai_configured": ai_manager.is_configured(),
        "current_model": ai_manager.current_model if ai_manager.is_configured() else None,
        "provider": ai_manager.current_provider if ai_manager.is_configured() else None,
        "available_features": [
            "chat",
            "analytics", 
            "session_management",
            "model_configuration"
        ]
    }

@router.get("/models/providers")
async def get_model_providers():
    """
    Get available model providers and their models.
    """
    try:
        models = ai_manager.get_available_models()
        providers = []
        
        for provider, model_list in models.items():
            provider_info = {
                "name": provider,
                "display_name": provider.title(),
                "models": []
            }
            
            for model in model_list:
                model_info = {
                    "id": model,
                    "name": model,
                    "description": f"{provider.title()} model: {model}"
                }
                provider_info["models"].append(model_info)
            
            providers.append(provider_info)
        
        return {
            "providers": providers,
            "current_provider": ai_manager.current_provider,
            "current_model": ai_manager.current_model
        }
    except Exception as e:
        logger.log_message(f"Error getting providers: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/status")
async def voice_status():
    """
    Voice functionality status (placeholder for future implementation).
    """
    return {
        "voice_enabled": False,
        "message": "Voice functionality not yet implemented",
        "supported_languages": ["es", "en"],
        "status": "disabled"
    }

@router.get("/search")
async def search_endpoint(q: str):
    """
    Search endpoint for web queries using Brave Search API.
    """
    try:
        if not q or len(q.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
        
        logger.log_message(f"Performing web search for: {q}", level=logging.INFO)
        
        # Get Brave Search API key from environment
        brave_api_key = os.getenv('BRAVE_SEARCH_API_KEY')
        
        if not brave_api_key:
            logger.log_message("No Brave Search API key configured", level=logging.WARNING)
            
            # Fallback to AI if no Brave API key
            if ai_manager.is_configured():
                messages = [{"role": "user", "content": f"Proporciona información actualizada sobre: {q}. Incluye fuentes si es posible. Nota: No tengo acceso a búsqueda web en tiempo real."}]
                response = ai_manager.generate_response(messages)
                
                return {
                    "query": q,
                    "results": [
                        {
                            "title": f"Información sobre: {q}",
                            "content": response,
                            "source": "AI Assistant",
                            "type": "ai_response"
                        }
                    ],
                    "total_results": 1,
                    "note": "Brave Search API key not configured, using AI response",
                    "search_engine": "AI Fallback"
                }
            else:
                raise HTTPException(status_code=500, detail="Brave Search API key not configured and AI not available")
        
        try:
            # Use Brave Search API
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': brave_api_key
            }
            
            # Detect language and set appropriate parameters
            is_spanish = any(spanish_word in q.lower() for spanish_word in ['méxico', 'mexico', 'aranceles', 'noticias', 'últimas', 'entre', 'usa', 'estados unidos'])
            
            params = {
                'q': q,
                'count': 5,
                'offset': 0,
                'country': 'MX' if is_spanish else 'US',
                'search_lang': 'es' if is_spanish else 'en',
                'ui_lang': 'es-MX' if is_spanish else 'en-US',
                'safesearch': 'moderate',
                'freshness': 'pd'  # Past day for recent news
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.search.brave.com/res/v1/web/search',
                    headers=headers,
                    params=params,
                    timeout=10.0
                )
            
            if response.status_code == 200:
                data = response.json()
                web_results = data.get('web', {}).get('results', [])
                
                # Format results
                formatted_results = []
                for result in web_results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "content": result.get("description", ""),
                        "url": result.get("url", ""),
                        "source": result.get("url", "").split("/")[2] if result.get("url") else "Unknown",
                        "type": "web_result",
                        "published": result.get("age", "")
                    })
                
                logger.log_message(f"Found {len(formatted_results)} search results from Brave", level=logging.INFO)
                
                return {
                    "query": q,
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                    "search_engine": "Brave Search"
                }
            else:
                logger.log_message(f"Brave Search API error: {response.status_code}", level=logging.WARNING)
                raise Exception(f"Brave Search API returned status {response.status_code}")
                
        except Exception as search_error:
            logger.log_message(f"Brave Search error: {str(search_error)}", level=logging.ERROR)
            
            # Fallback to AI on search error
            if ai_manager.is_configured():
                messages = [{"role": "user", "content": f"Proporciona información actualizada sobre: {q}. Nota: No pude acceder a búsqueda web en tiempo real debido a un error técnico, pero proporciona la mejor información que tengas disponible."}]
                response = ai_manager.generate_response(messages)
                
                return {
                    "query": q,
                    "results": [
                        {
                            "title": f"Información sobre: {q}",
                            "content": response,
                            "source": "AI Assistant",
                            "type": "ai_response"
                        }
                    ],
                    "total_results": 1,
                    "note": f"Brave Search failed: {str(search_error)}",
                    "search_engine": "AI Fallback"
                }
            else:
                raise HTTPException(status_code=500, detail=f"Brave Search failed: {str(search_error)}")
                
    except Exception as e:
        logger.log_message(f"Error in search endpoint: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback_data: dict):
    """
    Submit user feedback and ratings.
    """
    try:
        rating = feedback_data.get("rating", 0)
        feedback = feedback_data.get("feedback", "")
        
        # Log the feedback
        logger.log_message(f"User feedback received - Rating: {rating}/100, Feedback: {feedback}", level=logging.INFO)
        
        # Here you could save to database, send to analytics service, etc.
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.log_message(f"Error processing feedback: {str(e)}", level=logging.ERROR)
        return {"status": "error", "message": str(e)}

@router.post("/error")
async def error_handler(error_data: dict = None):
    """
    Error reporting endpoint.
    """
    try:
        if error_data:
            logger.log_message(f"Frontend error reported: {error_data}", level=logging.ERROR)
        
        return {
            "status": "error_logged",
            "message": "Error has been logged successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.log_message(f"Error in error handler: {str(e)}", level=logging.ERROR)
        return {"status": "error", "message": str(e)}

@router.post("/models/configure")
async def configure_model(config_data: dict):
    """
    Configure the AI model from frontend selection.
    """
    try:
        provider = config_data.get("provider")
        model = config_data.get("model")
        
        if not provider or not model:
            raise HTTPException(
                status_code=400, 
                detail="Both provider and model are required"
            )
        
        # Get the API key for the provider
        api_key = None
        if provider.lower() == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail=f"No API key found for provider: {provider}"
            )
        
        # Configure the model
        result = ai_manager.configure_model(provider, model, api_key)
        
        if result["status"] == "success":
            logger.log_message(f"Model configured successfully: {provider}/{model}", level=logging.INFO)
            return {
                "status": "success",
                "message": f"Model {model} from {provider} configured successfully",
                "provider": provider,
                "model": model
            }
        else:
            logger.log_message(f"Failed to configure model: {result.get('error')}", level=logging.ERROR)
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Failed to configure model")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error configuring model: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint to handle multiple file uploads from the frontend.
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            # Validate file size (max 10MB per file)
            file_content = await file.read()
            if len(file_content) > 10 * 1024 * 1024:  # 10MB
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} is too large. Maximum size is 10MB."
                )
            
            # Generate a unique filename
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(uploads_dir, unique_filename)
            
            # Save the file to disk
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_content)
            
            # Add file info to response
            uploaded_files.append({
                "original_name": file.filename,
                "saved_name": unique_filename,
                "path": file_path,
                "size": len(file_content),
                "content_type": file.content_type
            })
            
            logger.log_message(f"File uploaded: {file.filename} -> {unique_filename}", level=logging.INFO)
        
        return {
            "status": "success",
            "message": f"{len(uploaded_files)} file(s) uploaded successfully",
            "uploaded_files": uploaded_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error in upload: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/files")
async def list_uploaded_files():
    """
    List all uploaded files.
    """
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            return {"files": []}
        
        files = []
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"files": files}
        
    except Exception as e:
        logger.log_message(f"Error listing files: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/files/{filename}")
async def download_file(filename: str):
    """
    Download a specific uploaded file.
    """
    try:
        uploads_dir = "uploads"
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Return file as streaming response
        async def file_generator():
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_generator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error downloading file: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/csv/{filename}")
async def analyze_csv_endpoint(filename: str):
    """
    Analyze a specific CSV file and return EDA results.
    """
    try:
        uploads_dir = "uploads"
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        analysis = analyze_csv_file(file_path)
        
        if 'error' in analysis:
            raise HTTPException(status_code=500, detail=analysis['error'])
        
        return {
            "filename": filename,
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error analyzing CSV: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute/python")
async def execute_python_code(code_data: dict):
    """
    Execute Python code safely (fallback for when Pyodide is not available).
    This is a simplified version - in production you'd want proper sandboxing.
    """
    try:
        code = code_data.get("code", "")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="No code provided")
        
        # For security, we'll limit what can be executed
        # In a real implementation, you'd use a proper sandbox
        forbidden_imports = ['os', 'sys', 'subprocess', 'eval', 'exec', '__import__']
        
        for forbidden in forbidden_imports:
            if forbidden in code:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Forbidden operation: {forbidden}"
                )
        
        # Capture output
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        result = {
            "output": "",
            "error": None,
            "plots": [],
            "success": True
        }
        
        try:
            # Create a restricted globals environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'abs': abs,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'sorted': sorted,
                    'reversed': reversed,
                    'enumerate': enumerate,
                    'zip': zip,
                }
            }
            
            # Try to import safe libraries
            try:
                import numpy as np
                import pandas as pd
                safe_globals['np'] = np
                safe_globals['pd'] = pd
            except ImportError:
                pass
            
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, safe_globals)
            
            result["output"] = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            
            if error_output:
                result["error"] = error_output
                result["success"] = False
                
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error executing Python code: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/examples/python")
async def get_python_examples():
    """
    Get pre-built Python code examples for testing.
    """
    examples = {
        "basic_plot": {
            "title": "Gráfico de Líneas Básico",
            "description": "Un gráfico de líneas simple con datos de ejemplo",
            "code": """import matplotlib.pyplot as plt
import numpy as np

# Generar datos de ejemplo
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Función Seno', fontsize=16, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("Gráfico de función seno generado exitosamente!")"""
        },
        "bar_chart": {
            "title": "Gráfico de Barras con Estilo",
            "description": "Gráfico de barras colorido con datos de ventas",
            "code": """import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
productos = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E']
ventas = [23, 45, 56, 78, 32]
colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Crear gráfico de barras
plt.figure(figsize=(12, 8))
bars = plt.bar(productos, ventas, color=colores, alpha=0.8, edgecolor='black', linewidth=1.2)

# Personalizar el gráfico
plt.title('Ventas por Producto', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Productos', fontsize=14)
plt.ylabel('Ventas (unidades)', fontsize=14)
plt.xticks(rotation=45)

# Agregar valores en las barras
for bar, valor in zip(bars, ventas):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(valor), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Total de ventas: {sum(ventas)} unidades")
print(f"Producto más vendido: {productos[ventas.index(max(ventas))]} ({max(ventas)} unidades)")"""
        },
        "data_analysis": {
            "title": "Análisis de Datos con Pandas",
            "description": "Análisis estadístico básico con visualización",
            "code": """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear dataset de ejemplo
np.random.seed(42)
data = {
    'Edad': np.random.normal(35, 10, 1000),
    'Salario': np.random.normal(50000, 15000, 1000),
    'Experiencia': np.random.normal(8, 4, 1000),
    'Satisfaccion': np.random.uniform(1, 10, 1000)
}

df = pd.DataFrame(data)
df['Edad'] = df['Edad'].clip(18, 65).round()
df['Salario'] = df['Salario'].clip(20000, 100000).round()
df['Experiencia'] = df['Experiencia'].clip(0, 30).round()

# Análisis estadístico
print("=== ANÁLISIS ESTADÍSTICO ===")
print(df.describe())
print(f"\\nTamaño del dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

# Crear visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Dashboard de Análisis de Empleados', fontsize=16, fontweight='bold')

# Histograma de edades
axes[0,0].hist(df['Edad'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
axes[0,0].set_title('Distribución de Edades')
axes[0,0].set_xlabel('Edad')
axes[0,0].set_ylabel('Frecuencia')

# Scatter plot Salario vs Experiencia
axes[0,1].scatter(df['Experiencia'], df['Salario'], alpha=0.6, color='green')
axes[0,1].set_title('Salario vs Experiencia')
axes[0,1].set_xlabel('Años de Experiencia')
axes[0,1].set_ylabel('Salario ($)')

# Box plot de satisfacción
axes[1,0].boxplot(df['Satisfaccion'])
axes[1,0].set_title('Distribución de Satisfacción')
axes[1,0].set_ylabel('Nivel de Satisfacción (1-10)')

# Correlación
correlation = df[['Edad', 'Salario', 'Experiencia', 'Satisfaccion']].corr()
im = axes[1,1].imshow(correlation, cmap='coolwarm', aspect='auto')
axes[1,1].set_title('Matriz de Correlación')
axes[1,1].set_xticks(range(len(correlation.columns)))
axes[1,1].set_yticks(range(len(correlation.columns)))
axes[1,1].set_xticklabels(correlation.columns, rotation=45)
axes[1,1].set_yticklabels(correlation.columns)

# Agregar valores de correlación
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        text = axes[1,1].text(j, i, f'{correlation.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

print("\\n=== INSIGHTS CLAVE ===")
print(f"Edad promedio: {df['Edad'].mean():.1f} años")
print(f"Salario promedio: ${df['Salario'].mean():,.0f}")
print(f"Experiencia promedio: {df['Experiencia'].mean():.1f} años")
print(f"Satisfacción promedio: {df['Satisfaccion'].mean():.1f}/10")"""
        },
        "advanced_visualization": {
            "title": "Visualización Avanzada con Seaborn",
            "description": "Gráficos avanzados con múltiples variables",
            "code": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Crear dataset más complejo
np.random.seed(42)
n_samples = 500

data = {
    'Ventas': np.random.gamma(2, 2, n_samples) * 1000,
    'Marketing': np.random.exponential(2, n_samples) * 500,
    'Region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n_samples),
    'Trimestre': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n_samples),
    'Categoria': np.random.choice(['Premium', 'Standard', 'Básico'], n_samples)
}

df = pd.DataFrame(data)

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(16, 12))

# 1. Violin plot por región
plt.subplot(2, 3, 1)
sns.violinplot(data=df, x='Region', y='Ventas')
plt.title('Distribución de Ventas por Región', fontweight='bold')
plt.xticks(rotation=45)

# 2. Heatmap de ventas promedio
plt.subplot(2, 3, 2)
pivot_data = df.groupby(['Region', 'Trimestre'])['Ventas'].mean().unstack()
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Ventas Promedio: Región vs Trimestre', fontweight='bold')

# 3. Scatter plot con categorías
plt.subplot(2, 3, 3)
for categoria in df['Categoria'].unique():
    subset = df[df['Categoria'] == categoria]
    plt.scatter(subset['Marketing'], subset['Ventas'], 
               label=categoria, alpha=0.6, s=50)
plt.xlabel('Inversión en Marketing')
plt.ylabel('Ventas')
plt.title('Marketing vs Ventas por Categoría', fontweight='bold')
plt.legend()

# 4. Box plot múltiple
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='Categoria', y='Ventas', hue='Region')
plt.title('Ventas por Categoría y Región', fontweight='bold')
plt.xticks(rotation=45)

# 5. Histograma con KDE
plt.subplot(2, 3, 5)
for categoria in df['Categoria'].unique():
    subset = df[df['Categoria'] == categoria]['Ventas']
    plt.hist(subset, alpha=0.5, label=categoria, bins=20, density=True)
    sns.kdeplot(subset, label=f'{categoria} KDE')
plt.xlabel('Ventas')
plt.ylabel('Densidad')
plt.title('Distribución de Ventas con KDE', fontweight='bold')
plt.legend()

# 6. Gráfico de barras agrupadas
plt.subplot(2, 3, 6)
summary = df.groupby(['Region', 'Categoria'])['Ventas'].mean().unstack()
summary.plot(kind='bar', ax=plt.gca())
plt.title('Ventas Promedio por Región y Categoría', fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title='Categoría')

plt.tight_layout()
plt.show()

# Estadísticas resumidas
print("=== RESUMEN EJECUTIVO ===")
print(f"Total de registros analizados: {len(df):,}")
print(f"Ventas totales: ${df['Ventas'].sum():,.0f}")
print(f"Inversión total en marketing: ${df['Marketing'].sum():,.0f}")
print(f"ROI promedio: {(df['Ventas'].sum() / df['Marketing'].sum()):.2f}x")

print("\\n=== TOP PERFORMERS ===")
top_region = df.groupby('Region')['Ventas'].mean().idxmax()
top_categoria = df.groupby('Categoria')['Ventas'].mean().idxmax()
print(f"Mejor región: {top_region}")
print(f"Mejor categoría: {top_categoria}")"""
        },
        "time_series": {
            "title": "Análisis de Series Temporales",
            "description": "Visualización de datos temporales con tendencias",
            "code": """import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generar serie temporal
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365)]

# Crear tendencia con estacionalidad y ruido
trend = np.linspace(100, 150, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # 4 ciclos por año
noise = np.random.normal(0, 5, 365)
values = trend + seasonal + noise

# Crear DataFrame
df = pd.DataFrame({
    'Fecha': dates,
    'Valor': values
})

# Calcular medias móviles
df['MA_7'] = df['Valor'].rolling(window=7).mean()
df['MA_30'] = df['Valor'].rolling(window=30).mean()

# Crear visualización
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Análisis de Series Temporales - Ventas Diarias 2023', fontsize=16, fontweight='bold')

# Gráfico principal
axes[0].plot(df['Fecha'], df['Valor'], alpha=0.3, color='gray', label='Datos diarios')
axes[0].plot(df['Fecha'], df['MA_7'], color='blue', linewidth=2, label='Media móvil 7 días')
axes[0].plot(df['Fecha'], df['MA_30'], color='red', linewidth=2, label='Media móvil 30 días')
axes[0].set_title('Serie Temporal con Medias Móviles')
axes[0].set_ylabel('Ventas')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Análisis mensual
df['Mes'] = df['Fecha'].dt.month
monthly_avg = df.groupby('Mes')['Valor'].mean()
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

axes[1].bar(range(1, 13), monthly_avg.values, color='lightcoral', alpha=0.7)
axes[1].set_title('Promedio Mensual de Ventas')
axes[1].set_xlabel('Mes')
axes[1].set_ylabel('Ventas Promedio')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(meses)
axes[1].grid(True, alpha=0.3)

# Distribución de valores
axes[2].hist(df['Valor'], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[2].axvline(df['Valor'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["Valor"].mean():.1f}')
axes[2].axvline(df['Valor'].median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {df["Valor"].median():.1f}')
axes[2].set_title('Distribución de Valores')
axes[2].set_xlabel('Valor')
axes[2].set_ylabel('Frecuencia')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estadísticas
print("=== ANÁLISIS ESTADÍSTICO ===")
print(f"Período analizado: {df['Fecha'].min().strftime('%d/%m/%Y')} - {df['Fecha'].max().strftime('%d/%m/%Y')}")
print(f"Total de días: {len(df)}")
print(f"Valor promedio: {df['Valor'].mean():.2f}")
print(f"Desviación estándar: {df['Valor'].std():.2f}")
print(f"Valor mínimo: {df['Valor'].min():.2f}")
print(f"Valor máximo: {df['Valor'].max():.2f}")

print("\\n=== TENDENCIAS ===")
primer_trimestre = df[df['Fecha'].dt.quarter == 1]['Valor'].mean()
ultimo_trimestre = df[df['Fecha'].dt.quarter == 4]['Valor'].mean()
crecimiento = ((ultimo_trimestre - primer_trimestre) / primer_trimestre) * 100

print(f"Q1 promedio: {primer_trimestre:.2f}")
print(f"Q4 promedio: {ultimo_trimestre:.2f}")
print(f"Crecimiento anual: {crecimiento:.1f}%")"""
        }
    }
    
    return {"examples": examples}

@router.get("/files/available")
async def get_available_files():
    """
    Get information about all available uploaded files.
    """
    try:
        files_info = get_uploaded_files_info()
        return {
            "files": files_info,
            "total_files": len(files_info),
            "csv_files": len([f for f in files_info if f['extension'] == '.csv']),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.log_message(f"Error getting available files: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/web")
async def web_search_endpoint(search_data: dict):
    """
    Web search endpoint using the Brave Search API through the web search agent.
    """
    try:
        query = search_data.get("query", "")
        context = search_data.get("context", "")
        max_results = search_data.get("max_results", 10)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        # Get web search agent
        web_search_agent = get_web_search_agent()
        
        # Perform search
        search_results = web_search_agent.search(
            query=query,
            context=context,
            max_results=max_results
        )
        
        logger.log_message(f"Web search completed for query: {query}", level=logging.INFO)
        
        return {
            "status": "success",
            "search_results": search_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error in web search endpoint: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/data-science")
async def data_science_search_endpoint(search_data: dict):
    """
    Specialized search endpoint for data science topics.
    """
    try:
        topic = search_data.get("topic", "")
        specific_need = search_data.get("specific_need", "")
        
        if not topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")
        
        # Get web search agent
        web_search_agent = get_web_search_agent()
        
        # Perform specialized data science search
        search_results = web_search_agent.search_for_data_science(
            topic=topic,
            specific_need=specific_need
        )
        
        logger.log_message(f"Data science search completed for topic: {topic}", level=logging.INFO)
        
        return {
            "status": "success",
            "search_results": search_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.log_message(f"Error in data science search endpoint: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/datasets")
async def dataset_search_endpoint(search_data: dict):
    """
    Search for datasets related to a specific topic.
    """
    try:
        topic = search_data.get("topic", "")
        
        if not topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")
        
        # Get web search agent
        web_search_agent = get_web_search_agent()
        
        # Search for datasets
        search_results = web_search_agent.search_for_datasets(topic=topic)
        
        logger.log_message(f"Dataset search completed for topic: {topic}", level=logging.INFO)
        
        return {
            "status": "success",
            "search_results": search_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.log_message(f"Error in dataset search endpoint: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

def extract_python_code_blocks(text: str) -> List[str]:
    """
    Extract Python code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        
    Returns:
        List of Python code strings
    """
    # Pattern to match ```python code blocks
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def enhance_response_with_code_execution(response_text: str, data_context: str = "") -> str:
    """
    Enhance AI response by executing Python code blocks and adding insights.
    
    Args:
        response_text: Original AI response
        data_context: Context about the data being analyzed
        
    Returns:
        Enhanced response with execution results and insights
    """
    try:
        # Extract Python code blocks
        code_blocks = extract_python_code_blocks(response_text)
        
        if not code_blocks:
            return response_text
        
        # Only execute if there are 3 or fewer code blocks to avoid overwhelming the user
        if len(code_blocks) > 3:
            logger.log_message(f"Too many code blocks ({len(code_blocks)}), skipping auto-execution", level=logging.INFO)
            return response_text
        
        enhanced_response = response_text
        code_executor = get_code_executor()
        
        # Combine all code blocks into one execution for better flow
        combined_code = "\n\n".join(code_blocks)
        logger.log_message(f"Executing combined code blocks ({len(code_blocks)} blocks)", level=logging.INFO)
        
        # Execute and analyze the combined code
        execution_result = execute_and_analyze_code(combined_code, data_context)
        
        # Create a single execution results section at the end
        execution_section = f"\n\n---\n\n## 🚀 Resultados de Ejecución\n\n"
        
        if execution_result['success']:
            # Add text output if any
            if execution_result['output']:
                execution_section += f"### 📊 Salida del Código:\n```\n{execution_result['output']}\n```\n\n"
            
            # Add plots if any
            if execution_result['plots']:
                execution_section += f"### 📈 Visualizaciones Generadas: {len(execution_result['plots'])}\n"
                for j, plot_data in enumerate(execution_result['plots']):
                    execution_section += f"![Gráfico {j+1}](data:image/png;base64,{plot_data})\n\n"
            
            # Add AI-generated insights
            if execution_result['insights']:
                execution_section += f"### 🧠 Análisis e Insights:\n{execution_result['insights']}\n\n"
            
            # If no output, plots, or insights, just confirm execution
            if not execution_result['output'] and not execution_result['plots'] and not execution_result['insights']:
                execution_section += "✅ **Código ejecutado exitosamente**\n\n"
            
        else:
            # Add error information
            execution_section += f"### ❌ Error en la Ejecución:\n```\n{execution_result['error']}\n```\n\n"
            execution_section += "💡 **Sugerencia:** Puedes editar el código usando el botón 'Editar' para corregir errores.\n\n"
        
        # Append execution results at the end instead of inserting after each block
        enhanced_response = enhanced_response + execution_section
        
        return enhanced_response
        
    except Exception as e:
        logger.log_message(f"Error enhancing response with code execution: {str(e)}", level=logging.ERROR)
        return response_text  # Return original response if enhancement fails

@router.post("/validate/python")
async def validate_python_code(code_data: dict):
    """
    Validate Python code for syntax errors and formatting issues.
    
    Args:
        code_data: Dictionary containing 'code' field with Python code to validate
        
    Returns:
        Validation results with errors, warnings, and suggestions
    """
    try:
        code = code_data.get('code', '')
        
        if not code.strip():
            return {
                "valid": False,
                "errors": ["Código vacío"],
                "warnings": [],
                "suggestions": []
            }
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check for syntax errors
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Error de sintaxis en línea {e.lineno}: {e.msg}")
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Error de compilación: {str(e)}")
        
        # Check for common formatting issues
        lines = code.split('\n')
        
        # Check for concatenated statements
        for i, line in enumerate(lines, 1):
            if ')' in line and re.search(r'\)[a-zA-Z_]', line):
                validation_result["warnings"].append(f"Línea {i}: Posibles declaraciones concatenadas")
        
        # Check for broken file paths
        if re.search(r'uploads\s*\/\s*[a-f0-9]+\s*-', code):
            validation_result["warnings"].append("Rutas de archivo con espacios detectadas")
            validation_result["suggestions"].append("Usar rutas sin espacios: 'uploads/filename.csv'")
        
        # Check for broken operators
        if re.search(r'=\s*=\s*=', code) and '===' not in code:
            validation_result["warnings"].append("Operadores rotos detectados (= = =)")
            validation_result["suggestions"].append("Usar '===' para comentarios o '==' para comparaciones")
        
        # Check for missing imports
        has_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
        has_data_science_code = bool(re.search(r'\b(pd\.|np\.|plt\.|sns\.)', code))
        
        if has_data_science_code and not has_imports:
            validation_result["warnings"].append("Faltan declaraciones de import")
            validation_result["suggestions"].append("Agregar: import pandas as pd, numpy as np, matplotlib.pyplot as plt")
        
        # Check for broken comments
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not '=' in stripped and not '(' in stripped:
                if re.match(r'^[A-Z][a-z]+\s+(de|del|por|para|con|y|o)', stripped):
                    validation_result["warnings"].append(f"Línea {i}: Posible comentario sin #")
                    validation_result["suggestions"].append(f"Agregar # al inicio de la línea {i}")
        
        # Check for proper indentation
        for i, line in enumerate(lines, 1):
            if line.startswith(' ') and not line.startswith('    '):
                if len(line) - len(line.lstrip()) not in [0, 4, 8, 12, 16]:
                    validation_result["warnings"].append(f"Línea {i}: Indentación inconsistente")
                    validation_result["suggestions"].append("Usar 4 espacios para indentación")
        
        return validation_result
        
    except Exception as e:
        logger.log_message(f"Error validating Python code: {str(e)}", level=logging.ERROR)
        return {
            "valid": False,
            "errors": [f"Error interno de validación: {str(e)}"],
            "warnings": [],
            "suggestions": []
        }

@router.post("/format/python")
async def format_python_code_endpoint(code_data: dict):
    """
    Format Python code to fix common issues.
    
    Args:
        code_data: Dictionary containing 'code' field with Python code to format
        
    Returns:
        Formatted code with improvements applied
    """
    try:
        code = code_data.get('code', '')
        
        if not code.strip():
            return {
                "formatted_code": code,
                "changes_made": []
            }
        
        changes_made = []
        formatted_code = code
        
        # Fix broken file paths
        original_formatted = formatted_code
        formatted_code = re.sub(
            r'uploads\s*\/\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*-\s*([a-f0-9]+)\s*\.\s*csv',
            r'uploads/\1-\2-\3-\4-\5.csv',
            formatted_code
        )
        if formatted_code != original_formatted:
            changes_made.append("Corregidas rutas de archivo con espacios")
        
        # Fix broken operators
        original_formatted = formatted_code
        formatted_code = re.sub(r'=\s*=\s*=', '===', formatted_code)
        if formatted_code != original_formatted:
            changes_made.append("Corregidos operadores rotos")
        
        # Fix concatenated statements
        original_formatted = formatted_code
        formatted_code = re.sub(r'(\))([a-zA-Z_])', r'\1\n\2', formatted_code)
        if formatted_code != original_formatted:
            changes_made.append("Separadas declaraciones concatenadas")
        
        # Add missing imports if needed
        lines = formatted_code.split('\n')
        has_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
        has_data_science_code = bool(re.search(r'\b(pd\.|np\.|plt\.|sns\.)', formatted_code))
        
        if has_data_science_code and not has_imports:
            imports = [
                "import pandas as pd",
                "import numpy as np", 
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                ""
            ]
            formatted_code = '\n'.join(imports) + formatted_code
            changes_made.append("Agregadas declaraciones de import faltantes")
        
        # Fix broken comments
        lines = formatted_code.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not '=' in stripped and not '(' in stripped:
                if re.match(r'^[A-Z][a-z]+\s+(de|del|por|para|con|y|o)', stripped):
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(' ' * indent + '# ' + stripped)
                    if "Corregidos comentarios sin #" not in changes_made:
                        changes_made.append("Corregidos comentarios sin #")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        formatted_code = '\n'.join(fixed_lines)
        
        return {
            "formatted_code": formatted_code,
            "changes_made": changes_made
        }
        
    except Exception as e:
        logger.log_message(f"Error formatting Python code: {str(e)}", level=logging.ERROR)
        return {
            "formatted_code": code,
            "changes_made": [],
            "error": f"Error al formatear código: {str(e)}"
        } 