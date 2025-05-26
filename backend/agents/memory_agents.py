import dspy
from typing import Dict, Any, List
import logging
from utils.logger import Logger

logger = Logger("memory_agents", see_time=True, console_log=False)

class memory_agent(dspy.Module):
    """
    Memory agent that maintains conversation context and history.
    Helps other agents understand previous interactions and maintain continuity.
    """
    
    def __init__(self):
        super().__init__()
        self.memory_store = {}
        self.max_memory_items = 10  # Keep last 10 interactions
        
    def forward(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant memory context for the current query.
        
        Args:
            query: Current user query
            
        Returns:
            Dictionary containing relevant memory context
        """
        try:
            # Simple memory retrieval - in practice this could be more sophisticated
            memory_context = self._get_relevant_context(query)
            
            return {
                "memory_context": memory_context,
                "status": "success"
            }
        except Exception as e:
            logger.log_message(f"Error in memory_agent: {str(e)}", level=logging.ERROR)
            return {
                "memory_context": "",
                "status": "error",
                "error": str(e)
            }
    
    def update_memory(self, query: str, response: str):
        """
        Update memory with new interaction.
        
        Args:
            query: User query
            response: Agent response
        """
        try:
            # Create memory entry
            memory_entry = {
                "query": query,
                "response": response[:500],  # Truncate long responses
                "timestamp": self._get_timestamp()
            }
            
            # Add to memory store
            if "interactions" not in self.memory_store:
                self.memory_store["interactions"] = []
            
            self.memory_store["interactions"].append(memory_entry)
            
            # Keep only recent interactions
            if len(self.memory_store["interactions"]) > self.max_memory_items:
                self.memory_store["interactions"] = self.memory_store["interactions"][-self.max_memory_items:]
                
        except Exception as e:
            logger.log_message(f"Error updating memory: {str(e)}", level=logging.ERROR)
    
    def _get_relevant_context(self, query: str) -> str:
        """
        Get relevant context from memory based on current query.
        
        Args:
            query: Current query
            
        Returns:
            Relevant context string
        """
        if "interactions" not in self.memory_store or not self.memory_store["interactions"]:
            return "No previous context available."
        
        # Simple approach - return last few interactions
        recent_interactions = self.memory_store["interactions"][-3:]
        
        context_parts = []
        for interaction in recent_interactions:
            context_parts.append(f"Previous: {interaction['query']} -> {interaction['response'][:100]}...")
        
        return "\n".join(context_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        import datetime
        return datetime.datetime.now().isoformat()

class memory_summarize_agent(dspy.Module):
    """
    Agent that summarizes conversation history and memory.
    Helps compress long conversations into key insights.
    """
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(self._create_summary_signature())
    
    def _create_summary_signature(self):
        """Create the signature for memory summarization."""
        
        class MemorySummarySignature(dspy.Signature):
            """Summarize conversation history into key insights and context."""
            conversation_history = dspy.InputField(desc="Full conversation history to summarize")
            summary = dspy.OutputField(desc="Concise summary of key insights and context")
        
        return MemorySummarySignature
    
    def forward(self, conversation_history: str) -> Dict[str, Any]:
        """
        Summarize conversation history.
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Dictionary containing summary
        """
        try:
            if not conversation_history or conversation_history.strip() == "":
                return {
                    "summary": "No conversation history to summarize.",
                    "status": "success"
                }
            
            # Generate summary
            result = self.summarizer(conversation_history=conversation_history)
            
            return {
                "summary": result.summary,
                "status": "success"
            }
            
        except Exception as e:
            logger.log_message(f"Error in memory_summarize_agent: {str(e)}", level=logging.ERROR)
            return {
                "summary": "Error generating summary.",
                "status": "error",
                "error": str(e)
            } 