import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from .document_retrievers import DocumentRetriever, SemanticRetriever

logger = logging.getLogger(__name__)

class AgentMemoryRetriever:
    """Retriever for agent memory records."""
    
    def __init__(self, memory_dir: str = None, use_semantic: bool = True):
        """
        Initialize the agent memory retriever.
        
        Args:
            memory_dir: Directory containing agent memory files
            use_semantic: Whether to use semantic retrieval (vs. keyword)
        """
        self.memory_dir = memory_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/agent_memory")
        self.use_semantic = use_semantic
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Create retrievers for each agent
        self.agent_retrievers = {}
        self._load_agent_memories()
    
    def _load_agent_memories(self):
        """Load memory files for all agents."""
        if not os.path.exists(self.memory_dir):
            return
        
        for file in os.listdir(self.memory_dir):
            if file.endswith('.json'):
                agent_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.memory_dir, file)
                self._load_agent_memory(agent_name, file_path)
    
    def _load_agent_memory(self, agent_name: str, memory_file: str):
        """Load memory for a specific agent."""
        try:
            with open(memory_file, 'r') as f:
                memories = json.load(f)
            
            # Create a retriever for this agent
            if self.use_semantic:
                retriever = SemanticRetriever()
            else:
                from .document_retrievers import KeywordRetriever
                retriever = KeywordRetriever()
            
            # Add memories as documents
            for i, memory in enumerate(memories):
                # Format memory content
                content = self._format_memory_content(memory)
                
                # Add to retriever
                retriever.add_document(
                    doc_id=f"{agent_name}_{i}",
                    content=content,
                    metadata={
                        "agent": agent_name,
                        "timestamp": memory.get("timestamp", ""),
                        "memory_type": memory.get("type", "general"),
                        "original": memory
                    }
                )
            
            self.agent_retrievers[agent_name] = retriever
            logger.info(f"Loaded {len(memories)} memories for agent {agent_name}")
        except Exception as e:
            logger.error(f"Error loading memories for agent {agent_name}: {str(e)}")
    
    def _format_memory_content(self, memory: Dict[str, Any]) -> str:
        """Format a memory entry as searchable content."""
        content = []
        
        # Add all text fields
        for key, value in memory.items():
            if key in ["timestamp", "agent", "type"]:
                continue
            
            if isinstance(value, str):
                content.append(f"{key}: {value}")
            elif isinstance(value, (dict, list)):
                content.append(f"{key}: {json.dumps(value)}")
        
        return "\n".join(content)
    
    def retrieve(self, query: str, agent_name: Optional[str] = None, top_k: int = 5,
                time_period: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant agent memories.
        
        Args:
            query: The search query
            agent_name: Optional name of the agent to search (or None for all)
            top_k: Number of top results to return
            time_period: Optional time period filter (e.g., "last 24 hours", "2023-01-01 to 2023-01-31")
            
        Returns:
            List of relevant memory entries
        """
        results = []
        
        # Determine which agents to search
        if agent_name and agent_name in self.agent_retrievers:
            agents_to_search = [agent_name]
        else:
            agents_to_search = list(self.agent_retrievers.keys())
        
        # Retrieve memories from each agent
        for agent in agents_to_search:
            agent_results = self.agent_retrievers[agent].retrieve(query, top_k=top_k)
            results.extend(agent_results)
        
        # Filter by time period if specified
        if time_period:
            start_time, end_time = self._parse_time_period(time_period)
            if start_time or end_time:
                filtered_results = []
                for result in results:
                    timestamp_str = result.get("metadata", {}).get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if (not start_time or timestamp >= start_time) and (not end_time or timestamp <= end_time):
                                filtered_results.append(result)
                        except:
                            # If timestamp parsing fails, include the result anyway
                            filtered_results.append(result)
                    else:
                        # If no timestamp, include the result anyway
                        filtered_results.append(result)
                results = filtered_results
        
        # Sort results by relevance score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Truncate to top_k
        return results[:top_k]
    
    def _parse_time_period(self, time_period: str):
        """Parse a time period string into start and end times."""
        now = datetime.now()
        
        if time_period.lower() == "all":
            return None, None
        
        if time_period.lower() == "today":
            return datetime(now.year, now.month, now.day), None
        
        if time_period.lower() == "yesterday":
            yesterday = now - pd.Timedelta(days=1)
            return datetime(yesterday.year, yesterday.month, yesterday.day), datetime(now.year, now.month, now.day)
        
        if time_period.lower().startswith("last "):
            parts = time_period.lower().split()
            if len(parts) >= 3 and parts[1].isdigit():
                num = int(parts[1])
                unit = parts[2]
                if unit.startswith("hour"):
                    return now - pd.Timedelta(hours=num), None
                if unit.startswith("day"):
                    return now - pd.Timedelta(days=num), None
                if unit.startswith("week"):
                    return now - pd.Timedelta(weeks=num), None
                if unit.startswith("month"):
                    return now - pd.Timedelta(days=num*30), None  # Approximate
        
        if " to " in time_period:
            start_str, end_str = time_period.split(" to ")
            try:
                start_time = datetime.strptime(start_str, "%Y-%m-%d")
                end_time = datetime.strptime(end_str, "%Y-%m-%d")
                return start_time, end_time
            except:
                pass
        
        # Default: include all time periods
        return None, None
    
    def save_memory(self, agent_name: str, memory: Dict[str, Any]):
        """
        Save a new memory entry for an agent.
        
        Args:
            agent_name: Name of the agent
            memory: Memory data to save
        """
        # Ensure the memory has a timestamp
        if "timestamp" not in memory:
            memory["timestamp"] = datetime.now().isoformat()
        
        # Load existing memories
        memories = self._load_memories_for_agent(agent_name)
        
        # Add the new memory
        memories.append(memory)
        
        # Save to file
        self._save_memories_for_agent(agent_name, memories)
        
        # Update the retriever
        if agent_name not in self.agent_retrievers:
            if self.use_semantic:
                self.agent_retrievers[agent_name] = SemanticRetriever()
            else:
                from .document_retrievers import KeywordRetriever
                self.agent_retrievers[agent_name] = KeywordRetriever()
        
        # Add to retriever
        content = self._format_memory_content(memory)
        self.agent_retrievers[agent_name].add_document(
            doc_id=f"{agent_name}_{len(memories)-1}",
            content=content,
            metadata={
                "agent": agent_name,
                "timestamp": memory.get("timestamp", ""),
                "memory_type": memory.get("type", "general"),
                "original": memory
            }
        )
    
    def _load_memories_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Load all memories for a specific agent."""
        memory_file = os.path.join(self.memory_dir, f"{agent_name}.json")
        
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memories for agent {agent_name}: {str(e)}")
        
        return []
    
    def _save_memories_for_agent(self, agent_name: str, memories: List[Dict[str, Any]]):
        """Save all memories for a specific agent."""
        memory_file = os.path.join(self.memory_dir, f"{agent_name}.json")
        
        try:
            with open(memory_file, 'w') as f:
                json.dump(memories, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memories for agent {agent_name}: {str(e)}")


class ErrorMemoryRetriever:
    """Retriever for error memory records."""
    
    def __init__(self, error_db_path: str = None, use_semantic: bool = True):
        """
        Initialize the error memory retriever.
        
        Args:
            error_db_path: Path to the error database file
            use_semantic: Whether to use semantic retrieval (vs. keyword)
        """
        self.error_db_path = error_db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/error_memory.json")
        self.use_semantic = use_semantic
        self._ensure_error_db_exists()
        
        # Create retriever
        if self.use_semantic:
            self.retriever = SemanticRetriever()
        else:
            from .document_retrievers import KeywordRetriever
            self.retriever = KeywordRetriever()
        
        # Load error database
        self._load_error_db()
    
    def _ensure_error_db_exists(self):
        """Ensure the error database file exists."""
        os.makedirs(os.path.dirname(self.error_db_path), exist_ok=True)
        if not os.path.exists(self.error_db_path):
            with open(self.error_db_path, 'w') as f:
                json.dump([], f)
    
    def _load_error_db(self):
        """Load the error database into the retriever."""
        try:
            with open(self.error_db_path, 'r') as f:
                error_db = json.load(f)
            
            # Add errors as documents
            for i, error in enumerate(error_db):
                # Format error content
                content = self._format_error_content(error)
                
                # Add to retriever
                self.retriever.add_document(
                    doc_id=f"error_{i}",
                    content=content,
                    metadata={
                        "agent": error.get("agent", "UnknownAgent"),
                        "timestamp": error.get("timestamp", ""),
                        "error_type": error.get("analysis", {}).get("type", "Unknown"),
                        "severity": error.get("analysis", {}).get("severity", "Unknown"),
                        "original": error
                    }
                )
            
            logger.info(f"Loaded {len(error_db)} error records")
        except Exception as e:
            logger.error(f"Error loading error database: {str(e)}")
    
    def _format_error_content(self, error: Dict[str, Any]) -> str:
        """Format an error entry as searchable content."""
        content_parts = []
        
        # Add error message
        if "error_message" in error:
            content_parts.append(f"Error: {error['error_message']}")
        
        # Add traceback if available
        if "traceback" in error and error["traceback"]:
            content_parts.append(f"Traceback: {error['traceback']}")
        
        # Add analysis if available
        if "analysis" in error:
            analysis = error["analysis"]
            if "type" in analysis:
                content_parts.append(f"Type: {analysis['type']}")
            if "severity" in analysis:
                content_parts.append(f"Severity: {analysis['severity']}")
            if "root_cause" in analysis:
                content_parts.append(f"Root Cause: {analysis['root_cause']}")
        
        # Add context if available
        if "context" in error and error["context"]:
            content_parts.append(f"Context: {error['context']}")
        
        return "\n".join(content_parts)
    
    def retrieve(self, query: str, agent_name: Optional[str] = None, error_type: Optional[str] = None,
                severity: Optional[str] = None, time_period: Optional[str] = None, 
                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant error records.
        
        Args:
            query: The search query
            agent_name: Optional name of the agent to filter by
            error_type: Optional error type to filter by
            severity: Optional severity level to filter by
            time_period: Optional time period filter (e.g., "last 24 hours", "2023-01-01 to 2023-01-31")
            top_k: Number of top results to return
            
        Returns:
            List of relevant error records
        """
        # First retrieve based on the query
        results = self.retriever.retrieve(query, top_k=top_k*2)  # Get more initially for filtering
        
        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result.get("metadata", {})
            
            # Filter by agent name
            if agent_name and metadata.get("agent") != agent_name:
                continue
            
            # Filter by error type
            if error_type and metadata.get("error_type") != error_type:
                continue
            
            # Filter by severity
            if severity and metadata.get("severity") != severity:
                continue
            
            # Filter by time period
            if time_period:
                start_time, end_time = self._parse_time_period(time_period)
                if start_time or end_time:
                    timestamp_str = metadata.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if (start_time and timestamp < start_time) or (end_time and timestamp > end_time):
                                continue
                        except:
                            pass  # If timestamp parsing fails, include the result
            
            filtered_results.append(result)
        
        # Return top k filtered results
        return filtered_results[:top_k]
    
    def _parse_time_period(self, time_period: str):
        """Parse a time period string into start and end times."""
        now = datetime.now()
        
        if time_period.lower() == "all":
            return None, None
        
        if time_period.lower() == "today":
            return datetime(now.year, now.month, now.day), None
        
        if time_period.lower() == "yesterday":
            yesterday = now - pd.Timedelta(days=1)
            return datetime(yesterday.year, yesterday.month, yesterday.day), datetime(now.year, now.month, now.day)
        
        if time_period.lower().startswith("last "):
            parts = time_period.lower().split()
            if len(parts) >= 3 and parts[1].isdigit():
                num = int(parts[1])
                unit = parts[2]
                if unit.startswith("hour"):
                    return now - pd.Timedelta(hours=num), None
                if unit.startswith("day"):
                    return now - pd.Timedelta(days=num), None
                if unit.startswith("week"):
                    return now - pd.Timedelta(weeks=num), None
                if unit.startswith("month"):
                    return now - pd.Timedelta(days=num*30), None  # Approximate
        
        if " to " in time_period:
            start_str, end_str = time_period.split(" to ")
            try:
                start_time = datetime.strptime(start_str, "%Y-%m-%d")
                end_time = datetime.strptime(end_str, "%Y-%m-%d")
                return start_time, end_time
            except:
                pass
        
        # Default: include all time periods
        return None, None 