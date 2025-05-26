from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import os
import logging
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more files to the server.
    
    The files will be saved in the uploads directory with unique filenames.
    Returns information about the uploaded files including their paths.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    result = []
    for file in files:
        try:
            # Generate a unique filename to prevent overwriting
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_location = f"uploads/{unique_filename}"
            
            # Save the file
            file_content = await file.read()
            with open(file_location, "wb") as file_object:
                file_object.write(file_content)
            
            # Add file info to result
            file_info = {
                "original_name": file.filename,
                "saved_name": unique_filename,
                "path": file_location,
                "size": len(file_content),
                "content_type": file.content_type
            }
            result.append(file_info)
            
            logger.info(f"File uploaded: {file.filename} -> {file_location}")
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")
    
    return {"uploaded_files": result}

@router.get("/upload/list")
async def list_uploaded_files():
    """List all files that have been uploaded to the server."""
    try:
        files = []
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)
            if os.path.isfile(file_path):
                file_stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "path": file_path,
                    "size": file_stat.st_size,
                    "created": file_stat.st_ctime
                })
        
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing uploaded files: {str(e)}") 