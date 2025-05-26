"""
Global manager instances to avoid circular imports.
"""

from managers.ai_manager import AIManager
from managers.session_manager import SessionManager

# Create global instances
ai_manager = AIManager()
session_manager = SessionManager() 