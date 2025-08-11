# config.py
"""
Configuration settings for the Enhanced LLM Query Retrieval System
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "models/embedding-001"
    dimension: int = 768
    batch_size: int = 100
    fallback_to_tf: bool = True

@dataclass
class LLMConfig:
    """Configuration for Language Models."""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 30

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 50
    separators: list = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

@dataclass
class VectorDBConfig:
    """Configuration for vector databases."""
    default_method: str = "faiss"  # Options: faiss, chroma, pinecone, weaviate
    persist_directory: str = "./vector_db"
    collection_name_prefix: str = "hackrx_collection"
    similarity_threshold: float = 0.5
    max_results: int = 10

@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    default_k: int = 5
    search_type: str = "similarity"  # Options: similarity, mmr, similarity_score_threshold
    mmr_diversity_score: float = 0.5
    score_threshold: float = 0.7

@dataclass
class QAConfig:
    """Configuration for Q&A processing."""
    max_context_length: int = 4000
    answer_max_length: int = 500
    include_source_references: bool = True
    fallback_to_extraction: bool = True

class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        # Load environment variables
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = os.getenv("API_KEY", "your-secure-api-key")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Component configurations
        self.embedding = EmbeddingConfig()
        self.llm = LLMConfig()
        self.chunking = ChunkingConfig()
        self.vector_db = VectorDBConfig()
        self.retrieval = RetrievalConfig()
        self.qa = QAConfig()
        
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Document processing
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.allowed_extensions = ["pdf", "docx", "txt"]
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "300"))
        
        # Caching (if implemented)
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        if self.chunking.chunk_size < self.chunking.min_chunk_size:
            raise ValueError("Chunk size must be greater than minimum chunk size")
        
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.vector_db.default_method not in ["faiss", "chroma", "pinecone", "weaviate"]:
            raise ValueError("Invalid vector database method")
        
        return True
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get domain-specific prompt templates."""
        return {
            "insurance": """
You are an expert insurance analyst. Answer the question based on the insurance document context provided.
Focus on coverage details, policy terms, exclusions, deductibles, and claim procedures.

Context: {context}
Question: {question}

Provide a clear, accurate answer based only on the document content. If information is not available, state so explicitly.
""",
            
            "legal": """
You are a legal document specialist. Answer the question based on the legal document context provided.
Focus on legal requirements, obligations, rights, procedures, and compliance matters.

Context: {context}
Question: {question}

Provide a precise legal interpretation based only on the document content. Cite specific sections when possible.
""",
            
            "hr": """
You are an HR policy expert. Answer the question based on the HR document context provided.
Focus on employee policies, procedures, benefits, rights, responsibilities, and workplace guidelines.

Context: {context}
Question: {question}

Provide clear guidance based only on the document content. Reference specific policy sections when applicable.
""",
            
            "compliance": """
You are a compliance specialist. Answer the question based on the compliance document context provided.
Focus on regulatory requirements, audit procedures, risk management, and adherence to standards.

Context: {context}
Question: {question}

Provide detailed compliance guidance based only on the document content. Highlight any regulatory citations or requirements.
""",
            
            "general": """
You are an expert document analyst. Answer the question based on the provided context.
Provide accurate, comprehensive information based solely on the document content.

Context: {context}
Question: {question}

If the information is not available in the context, respond with: "The requested information is not available in the provided document."
"""
        }
    
    def get_vector_db_settings(self, method: str = None) -> Dict[str, Any]:
        """Get vector database specific settings."""
        method = method or self.vector_db.default_method
        
        settings = {
            "faiss": {
                "index_type": "IndexFlatIP",  # Inner Product for cosine similarity
                "metric": "cosine",
                "nlist": 100  # for IVF indices
            },
            
            "chroma": {
                "persist_directory": self.vector_db.persist_directory,
                "embedding_function": None,  # Will be set at runtime
                "collection_metadata": {"hnsw:space": "cosine"}
            },
            
            "pinecone": {
                "environment": os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": "hackrx-documents",
                "dimension": self.embedding.dimension,
                "metric": "cosine"
            },
            
            "weaviate": {
                "url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                "api_key": os.getenv("WEAVIATE_API_KEY"),
                "class_name": "Document",
                "distance_metric": "cosine"
            }
        }
        
        return settings.get(method, settings["faiss"])

# Global configuration instance
config = AppConfig()