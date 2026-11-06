"""Embedding services module for LLM Gateway.

Provides text embedding interfaces and implementations for local SentenceTransformers,
OpenAI, and Google Gemini with factory instantiation.
"""

from app.services.embeddings.base import BaseEmbeddingService
from app.services.embeddings.factory import EmbeddingServiceFactory
from app.services.embeddings.gemini_embed import GeminiEmbeddingService
from app.services.embeddings.local import LocalEmbeddingService
from app.services.embeddings.openai_embed import OpenAIEmbeddingService

__all__ = [
    "BaseEmbeddingService",
    "LocalEmbeddingService",
    "OpenAIEmbeddingService",
    "GeminiEmbeddingService",
    "EmbeddingServiceFactory",
]
