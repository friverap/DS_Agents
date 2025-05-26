from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import json
import os
from datetime import datetime
from typing import Optional

# Create feedback directory if it doesn't exist
os.makedirs("data/feedback", exist_ok=True)

router = APIRouter(
    prefix="/api",
    tags=["feedback"],
    responses={404: {"description": "Not found"}},
)

class FeedbackModel(BaseModel):
    rating: int
    feedback: Optional[str] = None

@router.post("/feedback")
async def submit_feedback(feedback_data: FeedbackModel):
    """
    Save user feedback and rating to a JSON file
    """
    try:
        # Generate a timestamp for the feedback
        timestamp = datetime.now().isoformat()
        
        # Create a feedback entry
        feedback_entry = {
            "timestamp": timestamp,
            "rating": feedback_data.rating,
            "feedback": feedback_data.feedback
        }
        
        # Create a unique filename based on timestamp
        filename = f"data/feedback/feedback_{timestamp.replace(':', '-')}.json"
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(feedback_entry, f, indent=4)
            
        # Also append to a running log of all feedback
        feedback_log_path = "data/feedback/feedback_log.json"
        
        # Check if the log file exists
        if os.path.exists(feedback_log_path):
            try:
                with open(feedback_log_path, "r") as f:
                    feedback_log = json.load(f)
            except json.JSONDecodeError:
                feedback_log = []
        else:
            feedback_log = []
        
        # Append new feedback
        feedback_log.append(feedback_entry)
        
        # Save updated log
        with open(feedback_log_path, "w") as f:
            json.dump(feedback_log, f, indent=4)
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save feedback: {str(e)}"
        ) 