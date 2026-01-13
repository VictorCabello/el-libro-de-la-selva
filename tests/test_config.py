"""Unit tests for el_libro_de_la_selva module configuration."""

import pytest
from el_libro_de_la_selva.config import Config


class TestConfig:
    """Test cases for Config dataclass."""

    def test_config_default_values(self):
        """Test that Config initializes with correct default values."""
        config = Config()
        assert config.LLM_MODEL == "llama3.2"
        assert config.LLM_TEMPERATURE == 0
        assert config.EMBEDDING_MODEL == "bge-m3"
        assert config.OLLAMA_URL == "http://localhost:11434"
        assert config.QDRANT_URL == "http://localhost:6333"
        assert config.INPUT_FILE == "El-libro-de-la-selva.txt"
        assert config.COLLECTION_NAME == "el-libro-de-la-selva"
        assert config.DEFAULT_CHUNK_SIZE == 1000
        assert config.CHUNK_OVERLAP == 200
        assert config.MAX_CLUSTERS == 5
        assert config.MAX_HIERARCHICAL_LAYERS == 3
        assert config.RANDOM_STATE == 42

    def test_config_custom_values(self):
        """Test that Config accepts and stores custom values."""
        config = Config(
            LLM_MODEL="custom-model",
            LLM_TEMPERATURE=1,
            EMBEDDING_MODEL="custom-embed",
            OLLAMA_URL="http://custom:11434",
            QDRANT_URL="http://custom:6333",
            INPUT_FILE="custom.txt",
            COLLECTION_NAME="custom-collection",
            DEFAULT_CHUNK_SIZE=2000,
            CHUNK_OVERLAP=400,
            MAX_CLUSTERS=10,
            MAX_HIERARCHICAL_LAYERS=5,
            RANDOM_STATE=100,
        )
        assert config.LLM_MODEL == "custom-model"
        assert config.LLM_TEMPERATURE == 1
        assert config.EMBEDDING_MODEL == "custom-embed"
        assert config.OLLAMA_URL == "http://custom:11434"
        assert config.QDRANT_URL == "http://custom:6333"
        assert config.INPUT_FILE == "custom.txt"
        assert config.COLLECTION_NAME == "custom-collection"
        assert config.DEFAULT_CHUNK_SIZE == 2000
        assert config.CHUNK_OVERLAP == 400
        assert config.MAX_CLUSTERS == 10
        assert config.MAX_HIERARCHICAL_LAYERS == 5
        assert config.RANDOM_STATE == 100

    def test_config_partial_custom_values(self):
        """Test that Config allows partial customization with defaults for other values."""
        config = Config(LLM_MODEL="custom-model", DEFAULT_CHUNK_SIZE=2000)
        assert config.LLM_MODEL == "custom-model"
        assert config.DEFAULT_CHUNK_SIZE == 2000
        assert config.LLM_TEMPERATURE == 0
        assert config.EMBEDDING_MODEL == "bge-m3"
