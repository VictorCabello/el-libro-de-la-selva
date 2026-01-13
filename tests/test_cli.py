"""Unit tests for CLI components."""

import pytest
from unittest.mock import MagicMock, patch
from el_libro_de_la_selva.cli.ingestor import main


class TestIngestorCLI:
    """Test cases for the ingestor CLI."""

    @patch('el_libro_de_la_selva.cli.ingestor.ChatOllama')
    @patch('el_libro_de_la_selva.cli.ingestor.OllamaEmbeddings')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentLoader')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentSplitter')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentClusterer')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentSummarizer')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentHierarchyBuilder')
    @patch('el_libro_de_la_selva.cli.ingestor.QdrantStorage')
    @patch('el_libro_de_la_selva.cli.ingestor.Config')
    def test_main_initializes_components(self, mock_config, mock_storage, mock_builder,
                                        mock_summarizer, mock_clusterer, mock_splitter,
                                        mock_loader, mock_embeddings, mock_llm):
        """Test that main function initializes all components correctly."""
        mock_config_instance = MagicMock()
        mock_config_instance.LLM_MODEL = "llama3.2"
        mock_config_instance.LLM_TEMPERATURE = 0
        mock_config_instance.EMBEDDING_MODEL = "bge-m3"
        mock_config_instance.OLLAMA_URL = "http://localhost:11434"
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.INPUT_FILE = "test.txt"
        mock_config_instance.COLLECTION_NAME = "test-collection"
        mock_config_instance.DEFAULT_CHUNK_SIZE = 1000
        mock_config_instance.CHUNK_OVERLAP = 200
        mock_config_instance.MAX_CLUSTERS = 5
        mock_config_instance.MAX_HIERARCHICAL_LAYERS = 3
        mock_config_instance.RANDOM_STATE = 42
        mock_config.return_value = mock_config_instance

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_from_file.return_value = [MagicMock(page_content="test")]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split.return_value = [MagicMock(page_content="chunk", metadata={"id": "1", "level": 1})]
        mock_splitter.return_value = mock_splitter_instance

        mock_builder_instance = MagicMock()
        mock_builder_instance.build_hierarchy.return_value = [MagicMock(page_content="hierarchy", metadata={"id": "1", "level": 1})]
        mock_builder.return_value = mock_builder_instance

        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance

        main()

        mock_llm.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_loader.assert_called_once()
        mock_splitter.assert_called_once()
        mock_clusterer.assert_called_once()
        mock_summarizer.assert_called_once()
        mock_builder.assert_called_once()
        mock_storage.assert_called_once()

    @patch('el_libro_de_la_selva.cli.ingestor.Config')
    @patch('el_libro_de_la_selva.cli.ingestor.DocumentLoader')
    def test_main_loads_document_from_config(self, mock_loader, mock_config):
        """Test that main loads document from file specified in Config."""
        mock_config_instance = MagicMock()
        mock_config_instance.INPUT_FILE = "test-file.txt"
        mock_config_instance.LLM_MODEL = "llama3.2"
        mock_config_instance.LLM_TEMPERATURE = 0
        mock_config_instance.EMBEDDING_MODEL = "bge-m3"
        mock_config_instance.OLLAMA_URL = "http://localhost:11434"
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.COLLECTION_NAME = "test-collection"
        mock_config_instance.DEFAULT_CHUNK_SIZE = 1000
        mock_config_instance.CHUNK_OVERLAP = 200
        mock_config_instance.MAX_CLUSTERS = 5
        mock_config_instance.MAX_HIERARCHICAL_LAYERS = 3
        mock_config_instance.RANDOM_STATE = 42
        mock_config.return_value = mock_config_instance

        with patch('el_libro_de_la_selva.cli.ingestor.ChatOllama'), \
             patch('el_libro_de_la_selva.cli.ingestor.OllamaEmbeddings'), \
             patch('el_libro_de_la_selva.cli.ingestor.DocumentSplitter'), \
             patch('el_libro_de_la_selva.cli.ingestor.DocumentClusterer'), \
             patch('el_libro_de_la_selva.cli.ingestor.DocumentSummarizer'), \
             patch('el_libro_de_la_selva.cli.ingestor.DocumentHierarchyBuilder'), \
             patch('el_libro_de_la_selva.cli.ingestor.QdrantStorage'):
            
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_from_file.return_value = [MagicMock(page_content="test")]
            mock_loader.return_value = mock_loader_instance

            main()

            mock_loader_instance.load_from_file.assert_called_once_with("test-file.txt")

    @patch('el_libro_de_la_selva.cli.ingestor.Config')
    def test_main_handles_file_not_found(self, mock_config):
        """Test that main handles FileNotFoundError appropriately."""
        mock_config_instance = MagicMock()
        mock_config_instance.INPUT_FILE = "nonexistent.txt"
        mock_config_instance.LLM_MODEL = "llama3.2"
        mock_config_instance.LLM_TEMPERATURE = 0
        mock_config_instance.EMBEDDING_MODEL = "bge-m3"
        mock_config_instance.OLLAMA_URL = "http://localhost:11434"
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.COLLECTION_NAME = "test-collection"
        mock_config_instance.DEFAULT_CHUNK_SIZE = 1000
        mock_config_instance.CHUNK_OVERLAP = 200
        mock_config_instance.MAX_CLUSTERS = 5
        mock_config_instance.MAX_HIERARCHICAL_LAYERS = 3
        mock_config_instance.RANDOM_STATE = 42
        mock_config.return_value = mock_config_instance

        with patch('el_libro_de_la_selva.cli.ingestor.ChatOllama'), \
             patch('el_libro_de_la_selva.cli.ingestor.OllamaEmbeddings'), \
             patch('el_libro_de_la_selva.cli.ingestor.DocumentLoader') as mock_loader:
            
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_from_file.side_effect = FileNotFoundError("File not found")
            mock_loader.return_value = mock_loader_instance

            with pytest.raises(FileNotFoundError):
                main()
