"""
Web Search Agent for DSAgency
Uses Brave Search API to find current information and research topics
"""

import os
import requests
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.logger import Logger

logger = Logger("web_search_agent", see_time=True, console_log=False)


class BraveSearchAPI:
    """
    Brave Search API client for web searches
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('BRAVE_SEARCH_API_KEY')
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        } if self.api_key else None
    
    def search(self, query: str, count: int = 10, country: str = "US", 
               search_lang: str = "en", ui_lang: str = "en-US") -> Dict[str, Any]:
        """
        Perform a web search using Brave Search API
        
        Args:
            query: Search query
            count: Number of results to return (max 20)
            country: Country code for search results
            search_lang: Language for search
            ui_lang: UI language
            
        Returns:
            Search results dictionary
        """
        if not self.api_key:
            logger.log_message("Brave Search API key not found", level=logging.WARNING)
            return self._fallback_search(query)
        
        try:
            params = {
                "q": query,
                "count": min(count, 20),  # Brave API max is 20
                "country": country,
                "search_lang": search_lang,
                "ui_lang": ui_lang,
                "safesearch": "moderate",
                "freshness": "pd",  # Past day for fresh results
                "text_decorations": False,
                "spellcheck": True
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.log_message(f"Brave Search successful for query: {query}", level=logging.INFO)
                return self._format_brave_results(data)
            else:
                logger.log_message(f"Brave Search API error: {response.status_code}", level=logging.ERROR)
                return self._fallback_search(query)
                
        except Exception as e:
            logger.log_message(f"Error in Brave Search: {str(e)}", level=logging.ERROR)
            return self._fallback_search(query)
    
    def _format_brave_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Brave Search API results into a standardized format
        """
        results = []
        
        # Process web results
        web_results = data.get("web", {}).get("results", [])
        for result in web_results:
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "published": result.get("age", ""),
                "source": "Brave Search"
            }
            results.append(formatted_result)
        
        # Process news results if available
        news_results = data.get("news", {}).get("results", [])
        for result in news_results:
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "published": result.get("age", ""),
                "source": "News"
            }
            results.append(formatted_result)
        
        return {
            "query": data.get("query", {}).get("original", ""),
            "results": results,
            "total_results": len(results),
            "search_time": datetime.utcnow().isoformat(),
            "api_used": "Brave Search"
        }
    
    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """
        Fallback search when Brave API is not available
        """
        logger.log_message(f"Using fallback search for: {query}", level=logging.INFO)
        
        # Simulate search results with helpful information
        fallback_results = [
            {
                "title": f"Search Results for: {query}",
                "url": "https://www.google.com/search?q=" + query.replace(" ", "+"),
                "description": "Brave Search API is not configured. Please set BRAVE_SEARCH_API_KEY environment variable to enable web search functionality.",
                "published": "N/A",
                "source": "Fallback"
            },
            {
                "title": "How to Configure Brave Search API",
                "url": "https://brave.com/search/api/",
                "description": "Get your free Brave Search API key to enable web search functionality in DSAgency.",
                "published": "N/A",
                "source": "Information"
            }
        ]
        
        return {
            "query": query,
            "results": fallback_results,
            "total_results": len(fallback_results),
            "search_time": datetime.utcnow().isoformat(),
            "api_used": "Fallback",
            "note": "Brave Search API not configured"
        }


class WebSearchAgent:
    """
    Web Search Agent that provides intelligent search capabilities
    """
    
    def __init__(self):
        self.brave_api = BraveSearchAPI()
        logger.log_message("Web Search Agent initialized", level=logging.INFO)
    
    def search(self, query: str, context: str = "", max_results: int = 10) -> Dict[str, Any]:
        """
        Perform an intelligent web search
        
        Args:
            query: Search query
            context: Additional context about the search
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results with insights
        """
        try:
            # Enhance query based on context
            enhanced_query = self._enhance_query(query, context)
            
            # Perform search
            search_results = self.brave_api.search(enhanced_query, count=max_results)
            
            # Add insights and summary
            search_results["insights"] = self._generate_insights(search_results)
            search_results["summary"] = self._generate_summary(search_results)
            
            return search_results
            
        except Exception as e:
            logger.log_message(f"Error in web search: {str(e)}", level=logging.ERROR)
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_time": datetime.utcnow().isoformat()
            }
    
    def _enhance_query(self, query: str, context: str) -> str:
        """
        Enhance search query based on context
        """
        # Add data science context if relevant
        data_science_keywords = [
            "machine learning", "data analysis", "statistics", "python",
            "pandas", "numpy", "matplotlib", "seaborn", "plotly", "sklearn"
        ]
        
        if any(keyword in query.lower() for keyword in data_science_keywords):
            if "python" not in query.lower():
                query += " python"
            if "tutorial" not in query.lower() and "how to" not in query.lower():
                query += " tutorial"
        
        # Add current year for recent information
        if "latest" in query.lower() or "current" in query.lower() or "2024" not in query:
            query += " 2024"
        
        return query
    
    def _generate_insights(self, search_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from search results
        """
        insights = []
        results = search_results.get("results", [])
        
        if not results:
            insights.append("No search results found. Try refining your search query.")
            return insights
        
        # Analyze result sources
        sources = [result.get("source", "Unknown") for result in results]
        unique_sources = set(sources)
        
        insights.append(f"Found {len(results)} results from {len(unique_sources)} different sources")
        
        # Check for recent results
        recent_count = sum(1 for result in results if "hour" in result.get("published", "") or "day" in result.get("published", ""))
        if recent_count > 0:
            insights.append(f"{recent_count} results are from recent sources (within 24 hours)")
        
        # Check for educational content
        educational_count = sum(1 for result in results if any(term in result.get("title", "").lower() for term in ["tutorial", "guide", "how to", "learn"]))
        if educational_count > 0:
            insights.append(f"{educational_count} results contain educational content")
        
        return insights
    
    def _generate_summary(self, search_results: Dict[str, Any]) -> str:
        """
        Generate a summary of search results
        """
        results = search_results.get("results", [])
        
        if not results:
            return "No relevant search results found."
        
        # Extract key themes from titles and descriptions
        all_text = " ".join([
            result.get("title", "") + " " + result.get("description", "")
            for result in results[:5]  # Use top 5 results
        ]).lower()
        
        # Common data science terms
        key_terms = {
            "machine learning": all_text.count("machine learning"),
            "data analysis": all_text.count("data analysis"),
            "python": all_text.count("python"),
            "visualization": all_text.count("visualization") + all_text.count("chart") + all_text.count("plot"),
            "statistics": all_text.count("statistics") + all_text.count("statistical"),
            "tutorial": all_text.count("tutorial") + all_text.count("guide")
        }
        
        # Find most common themes
        top_themes = sorted(key_terms.items(), key=lambda x: x[1], reverse=True)[:3]
        top_themes = [theme for theme, count in top_themes if count > 0]
        
        if top_themes:
            summary = f"Search results primarily focus on {', '.join(top_themes)}. "
        else:
            summary = "Search results cover various topics related to your query. "
        
        summary += f"Found {len(results)} relevant resources including documentation, tutorials, and current information."
        
        return summary
    
    def search_for_data_science(self, topic: str, specific_need: str = "") -> Dict[str, Any]:
        """
        Specialized search for data science topics
        """
        # Construct data science specific query
        query = f"{topic} data science python"
        
        if specific_need:
            query += f" {specific_need}"
        
        context = f"Data science search for {topic}"
        if specific_need:
            context += f" with focus on {specific_need}"
        
        return self.search(query, context)
    
    def search_for_current_trends(self, domain: str) -> Dict[str, Any]:
        """
        Search for current trends in a specific domain
        """
        query = f"latest trends {domain} 2024 current developments"
        context = f"Searching for current trends in {domain}"
        
        return self.search(query, context)
    
    def search_for_datasets(self, topic: str) -> Dict[str, Any]:
        """
        Search for datasets related to a topic
        """
        query = f"{topic} dataset download free kaggle github"
        context = f"Searching for datasets related to {topic}"
        
        return self.search(query, context)


# Global instance
web_search_agent = None

def get_web_search_agent():
    """Get or create the global web search agent instance"""
    global web_search_agent
    if web_search_agent is None:
        web_search_agent = WebSearchAgent()
    return web_search_agent 