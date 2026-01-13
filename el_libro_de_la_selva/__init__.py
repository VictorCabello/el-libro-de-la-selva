"""El Libro de la Selva - Document Ingestion and Retrieval System.

This module provides a comprehensive system for processing, organizing, and
retrieving documents using hierarchical clustering and vector search.

Key Features:
    - Hierarchical document processing with multi-level summarization
    - Vector-based storage using Qdrant for efficient similarity search
    - Flexible retrieval strategies (collapsed tree and tree traversal)
    - LangChain integration for seamless LLM and embedding operations
    - Local execution using Ollama for privacy and control

Main Components:
    - Config: Centralized configuration management
    - DocumentLoader: Load documents from text files
    - DocumentSplitter: Split documents into manageable chunks
    - DocumentClusterer: Cluster documents by embedding similarity
    - DocumentSummarizer: Generate summaries for document clusters
    - DocumentHierarchyBuilder: Build multi-layer document hierarchies
    - QdrantStorage: Store and retrieve documents from vector database
    - PromptTemplates: Reusable prompt templates for LLM operations
    - collapsed_tree_retrieval: Flat similarity search across all levels
    - tree_traversal_search: Hierarchical search following document structure

Example:
    >>> from langchain_ollama import OllamaEmbeddings, ChatOllama
    >>> from el_libro_de_la_selva import Config, DocumentLoader, DocumentSplitter
    >>>
    >>> # Initialize components
    >>> config = Config()
    >>> llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    >>> embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    >>>
    >>> # Load and split documents
    >>> loader = DocumentLoader()
    >>> docs = loader.load_from_file(config.INPUT_FILE)
    >>> splitter = DocumentSplitter(config.DEFAULT_CHUNK_SIZE, config.CHUNK_OVERLAP)
    >>> chunks = splitter.split(docs)

For detailed documentation, see DOCUMENTATION.md in the project root.
"""

from .config import Config
from .prompts import PromptTemplates
from .loader import DocumentLoader, DocumentSplitter
from .clustering import DocumentClusterer, DocumentSummarizer, DocumentHierarchyBuilder
from .storage import QdrantStorage
from .retrieval import collapsed_tree_retrieval, tree_traversal_search

__all__ = [
    "Config",
    "PromptTemplates",
    "DocumentLoader",
    "DocumentSplitter",
    "DocumentClusterer",
    "DocumentSummarizer",
    "DocumentHierarchyBuilder",
    "QdrantStorage",
    "collapsed_tree_retrieval",
    "tree_traversal_search",
]
