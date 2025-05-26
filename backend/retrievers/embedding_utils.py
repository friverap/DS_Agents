import os
import logging
import numpy as np
from typing import List, Union, Optional
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default OpenAI API key environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def embed_text(text: str, model: str = "openai") -> List[float]:
    """
    Generate embeddings for a text string.
    
    Args:
        text: The text to embed
        model: The embedding model to use (options: "openai", "huggingface")
        
    Returns:
        A list of float values representing the embedding
    """
    if model == "openai":
        return _embed_openai(text)
    elif model == "huggingface":
        return _embed_huggingface(text)
    elif model == "local":
        return _embed_local(text)
    else:
        logger.warning(f"Unknown embedding model: {model}, falling back to local embedding")
        return _embed_local(text)

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Convert to numpy arrays and reshape
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    return float(similarity)

def batch_embed_texts(texts: List[str], model: str = "openai") -> List[List[float]]:
    """
    Generate embeddings for multiple texts at once.
    
    Args:
        texts: List of text strings to embed
        model: The embedding model to use
        
    Returns:
        List of embedding vectors
    """
    if model == "openai":
        return _batch_embed_openai(texts)
    elif model == "huggingface":
        return [_embed_huggingface(text) for text in texts]
    elif model == "local":
        return [_embed_local(text) for text in texts]
    else:
        logger.warning(f"Unknown embedding model: {model}, falling back to local embedding")
        return [_embed_local(text) for text in texts]

def _embed_openai(text: str) -> List[float]:
    """Generate embeddings using OpenAI API."""
    try:
        import openai
        
        # Configure API key
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            return _embed_local(text)  # Fallback to local embedding
        
        # Make API request
        openai.api_key = api_key
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        # Extract embedding
        embedding = response.data[0].embedding
        return embedding
    
    except ImportError:
        logger.warning("OpenAI package not installed. Please install with `pip install openai`")
        return _embed_local(text)
    
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding: {str(e)}")
        return _embed_local(text)

def _batch_embed_openai(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts using OpenAI API."""
    try:
        import openai
        
        # Configure API key
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            return [_embed_local(text) for text in texts]  # Fallback to local embedding
        
        # Make API request
        openai.api_key = api_key
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    except ImportError:
        logger.warning("OpenAI package not installed. Please install with `pip install openai`")
        return [_embed_local(text) for text in texts]
    
    except Exception as e:
        logger.error(f"Error generating OpenAI embeddings: {str(e)}")
        return [_embed_local(text) for text in texts]

def _embed_huggingface(text: str) -> List[float]:
    """Generate embeddings using Hugging Face models."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model (will be cached after first load)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        return embedding
    
    except ImportError:
        logger.warning("sentence-transformers package not installed. Please install with `pip install sentence-transformers`")
        return _embed_local(text)
    
    except Exception as e:
        logger.error(f"Error generating Hugging Face embedding: {str(e)}")
        return _embed_local(text)

def _embed_local(text: str) -> List[float]:
    """
    Generate a simple locally-computed embedding for fallback.
    This is not meant for production use but as a fallback when other methods fail.
    """
    # Very simple embedding: just hash the text and use the hash values
    import hashlib
    
    # Create a hash of the text
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert hash bytes to floats between -1 and 1
    embedding = []
    for i in range(0, len(hash_bytes), 2):
        if i + 1 < len(hash_bytes):
            # Use pairs of bytes to create more varied values
            value = ((hash_bytes[i] << 8) + hash_bytes[i+1]) / 65535 * 2 - 1
            embedding.append(value)
    
    # Pad or truncate to fixed length (128 dimensions)
    embedding = embedding[:128] if len(embedding) > 128 else embedding + [0.0] * (128 - len(embedding))
    
    return embedding 