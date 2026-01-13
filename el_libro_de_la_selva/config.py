from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for the document ingestion and retrieval system.

    This dataclass provides centralized configuration management for all components
    of the el_libro_de_la_selva module, including LLM settings, embedding parameters,
    storage connections, and processing options.

    Attributes:
        LLM_MODEL: Name of the LLM model to use for summarization (default: "llama3.2")
        LLM_TEMPERATURE: Temperature for LLM generation, 0 for deterministic output (default: 0)
        EMBEDDING_MODEL: Name of the embedding model for vector generation (default: "bge-m3")
        OLLAMA_URL: Base URL for the Ollama API (default: "http://localhost:11434")
        QDRANT_URL: Base URL for the Qdrant vector database (default: "http://localhost:6333")
        INPUT_FILE: Path to the input text file for processing (default: "El-libro-de-la-selva.txt")
        COLLECTION_NAME: Name of the Qdrant collection (default: "el-libro-de-la-selva")
        DEFAULT_CHUNK_SIZE: Maximum characters per text chunk (default: 1000)
        CHUNK_OVERLAP: Overlap characters between adjacent chunks (default: 200)
        MAX_CLUSTERS: Maximum number of clusters per hierarchy layer (default: 5)
        MAX_HIERARCHICAL_LAYERS: Maximum depth of the document hierarchy (default: 3)
        RANDOM_STATE: Random seed for reproducible clustering (default: 42)
    """

    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: int = 0
    EMBEDDING_MODEL: str = "bge-m3"
    OLLAMA_URL: str = "http://localhost:11434"
    QDRANT_URL: str = "http://localhost:6333"
    INPUT_FILE: str = "El-libro-de-la-selva.txt"
    COLLECTION_NAME: str = "el-libro-de-la-selva"
    DEFAULT_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CLUSTERS: int = 5
    MAX_HIERARCHICAL_LAYERS: int = 3
    RANDOM_STATE: int = 42
