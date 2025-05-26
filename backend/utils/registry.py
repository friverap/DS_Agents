"""
Registry for agent classes.
"""
import logging
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)

# Store for agent classes
_agent_classes = {}

def register_agent_class(name: str, agent_class: type) -> None:
    """
    Register an agent class with the given name.
    
    Args:
        name: The name of the agent class
        agent_class: The agent class to register
    """
    logger.info(f"Registering agent class: {name}")
    _agent_classes[name] = agent_class

def get_agent_class(name: str) -> Optional[type]:
    """
    Get the agent class for the given name.
    
    Args:
        name: The name of the agent class
        
    Returns:
        The agent class, or None if not found
    """
    return _agent_classes.get(name)

def get_available_agent_classes() -> list:
    """
    Get a list of available agent class names.
    
    Returns:
        List of agent class names
    """
    return list(_agent_classes.keys()) 