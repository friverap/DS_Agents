import requests
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from config.settings import BRAVE_SEARCH_API_KEY

router = APIRouter(prefix="/api", tags=["search"])

class SearchResult(BaseModel):
    title: str
    url: str
    description: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@router.get("/search", response_model=SearchResponse)
async def web_search(
    q: str = Query(..., description="Search query"),
    count: Optional[int] = Query(10, description="Number of results to return")
):
    """
    Perform a web search using the Brave Search API.
    """
    if not BRAVE_SEARCH_API_KEY:
        raise HTTPException(status_code=500, detail="Brave Search API key is not configured")
        
    if not q:
        raise HTTPException(status_code=400, detail="Search query is required")
        
    # Brave Search API endpoint
    url = "https://api.search.brave.com/res/v1/web/search"
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY
    }
    
    params = {
        "q": q,
        "count": count
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Format the response
        formatted_results = {
            "query": q,
            "results": []
        }
        
        if "web" in search_results and "results" in search_results["web"]:
            for result in search_results["web"]["results"]:
                formatted_results["results"].append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", "")
                })
                
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error making request to Brave Search API: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing response from Brave Search API: {str(e)}") 