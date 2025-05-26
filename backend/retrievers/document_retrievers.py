import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_utils import embed_text, calculate_similarity

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Base class for document retrieval systems."""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the document retriever.
        
        Args:
            index_path: Optional path to a pre-built index file
        """
        self.documents = []
        self.document_index = {}
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the retriever.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata for the document
        """
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
        self._index_document(len(self.documents) - 1)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add multiple documents to the retriever.
        
        Args:
            documents: List of document dictionaries, each with 'id', 'content', and optional 'metadata'
        """
        start_idx = len(self.documents)
        self.documents.extend(documents)
        for i in range(start_idx, len(self.documents)):
            self._index_document(i)
    
    def _index_document(self, doc_idx: int):
        """
        Index a document for retrieval (to be implemented by subclasses).
        
        Args:
            doc_idx: Index of the document in the documents list
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query (to be implemented by subclasses).
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of the top k relevant documents
        """
        raise NotImplementedError("Subclasses must implement retrieve method")
    
    def save_index(self, path: str):
        """
        Save the document index to a file.
        
        Args:
            path: Path to save the index file
        """
        save_data = {
            "documents": self.documents,
            "index": self.document_index
        }
        with open(path, 'w') as f:
            json.dump(save_data, f)
    
    def load_index(self, path: str):
        """
        Load the document index from a file.
        
        Args:
            path: Path to the index file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        self.documents = data.get("documents", [])
        self.document_index = data.get("index", {})


class SemanticRetriever(DocumentRetriever):
    """Document retriever using semantic similarity for retrieval."""
    
    def __init__(self, index_path: Optional[str] = None, embedding_model: str = "openai"):
        """
        Initialize the semantic retriever.
        
        Args:
            index_path: Optional path to a pre-built index file
            embedding_model: Model to use for embeddings (default: openai)
        """
        super().__init__(index_path)
        self.embedding_model = embedding_model
        self.embeddings = []
    
    def _index_document(self, doc_idx: int):
        """
        Index a document by computing its embedding.
        
        Args:
            doc_idx: Index of the document in the documents list
        """
        document = self.documents[doc_idx]
        embedding = embed_text(document["content"], model=self.embedding_model)
        
        if len(self.embeddings) <= doc_idx:
            self.embeddings.append(embedding)
        else:
            self.embeddings[doc_idx] = embedding
        
        # Update the document index with the position of the embedding
        self.document_index[document["id"]] = doc_idx
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using semantic similarity.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of the top k semantically similar documents
        """
        if not self.documents:
            return []
        
        query_embedding = embed_text(query, model=self.embedding_model)
        
        # Calculate similarities
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = calculate_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top k indices
        if len(similarities) <= top_k:
            top_indices = list(range(len(similarities)))
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top k documents with their similarity scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": similarities[idx]
            })
        
        return results


class KeywordRetriever(DocumentRetriever):
    """Document retriever using keyword-based search."""
    
    def __init__(self, index_path: Optional[str] = None, case_sensitive: bool = False):
        """
        Initialize the keyword retriever.
        
        Args:
            index_path: Optional path to a pre-built index file
            case_sensitive: Whether to perform case-sensitive indexing and retrieval
        """
        super().__init__(index_path)
        self.case_sensitive = case_sensitive
        self.keyword_index = {}  # Maps keywords to document indices
    
    def _index_document(self, doc_idx: int):
        """
        Index a document by extracting keywords.
        
        Args:
            doc_idx: Index of the document in the documents list
        """
        document = self.documents[doc_idx]
        content = document["content"]
        
        if not self.case_sensitive:
            content = content.lower()
        
        # Simple tokenization - split by whitespace and punctuation
        words = ''.join(c if c.isalnum() else ' ' for c in content).split()
        
        # Add document to the keyword index
        for word in words:
            if word not in self.keyword_index:
                self.keyword_index[word] = []
            if doc_idx not in self.keyword_index[word]:
                self.keyword_index[word].append(doc_idx)
        
        # Update the document index
        self.document_index[document["id"]] = doc_idx
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using keyword matching.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of the top k documents matching the keywords in the query
        """
        if not self.documents:
            return []
        
        if not self.case_sensitive:
            query = query.lower()
        
        # Split query into keywords
        query_words = ''.join(c if c.isalnum() else ' ' for c in query).split()
        
        # Count occurrences of each document
        doc_counts = {}
        for word in query_words:
            if word in self.keyword_index:
                for doc_idx in self.keyword_index[word]:
                    doc_counts[doc_idx] = doc_counts.get(doc_idx, 0) + 1
        
        # Sort documents by their keyword match count
        sorted_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for doc_idx, count in sorted_docs[:top_k]:
            doc = self.documents[doc_idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": count / len(query_words)  # Normalize score
            })
        
        return results 