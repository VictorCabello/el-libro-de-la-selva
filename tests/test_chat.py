"""Unit tests for chat CLI component."""

import pytest
from unittest.mock import MagicMock, patch
from el_libro_de_la_selva.cli.chat import main


class TestChatCLI:
    """Test cases for the chat CLI."""

    @patch('el_libro_de_la_selva.cli.chat.OllamaEmbeddings')
    @patch('el_libro_de_la_selva.cli.chat.QdrantVectorStore')
    @patch('el_libro_de_la_selva.cli.chat.tree_traversal_search')
    @patch('el_libro_de_la_selva.cli.chat.Config')
    def test_main_performs_tree_traversal_search(self, mock_config, mock_search, mock_qdrant, mock_embeddings):
        """Test that main performs tree traversal search."""
        mock_config.EMBEDDING_MODEL = "bge-m3"
        mock_config.OLLAMA_URL = "http://localhost:11434"
        mock_config.QDRANT_URL = "http://localhost:6333"
        mock_config.COLLECTION_NAME = "test-collection"
        
        mock_search.return_value = [
            MagicMock(
                metadata={"level": 3},
                page_content="Summary content"
            )
        ]
        
        main()
        
        mock_qdrant.from_existing_collection.assert_called_once()
        mock_search.assert_called_once()

    @patch('el_libro_de_la_selva.cli.chat.OllamaEmbeddings')
    @patch('el_libro_de_la_selva.cli.chat.QdrantVectorStore')
    @patch('el_libro_de_la_selva.cli.chat.tree_traversal_search')
    @patch('el_libro_de_la_selva.cli.chat.Config')
    def test_main_prints_results(self, mock_config, mock_search, mock_qdrant, mock_embeddings, capsys):
        """Test that main prints search results."""
        mock_config.EMBEDDING_MODEL = "bge-m3"
        mock_config.OLLAMA_URL = "http://localhost:11434"
        mock_config.QDRANT_URL = "http://localhost:6333"
        mock_config.COLLECTION_NAME = "test-collection"
        
        mock_search.return_value = [
            MagicMock(
                metadata={"level": 3},
                page_content="Test content"
            )
        ]
        
        main()
        
        captured = capsys.readouterr()
        assert "Test content" in captured.out

    @patch('el_libro_de_la_selva.cli.chat.OllamaEmbeddings')
    @patch('el_libro_de_la_selva.cli.chat.QdrantVectorStore')
    @patch('el_libro_de_la_selva.cli.chat.tree_traversal_search')
    @patch('el_libro_de_la_selva.cli.chat.Config')
    def test_main_handles_empty_results(self, mock_config, mock_search, mock_qdrant, mock_embeddings, capsys):
        """Test that main handles empty search results."""
        mock_config.EMBEDDING_MODEL = "bge-m3"
        mock_config.OLLAMA_URL = "http://localhost:11434"
        mock_config.QDRANT_URL = "http://localhost:6333"
        mock_config.COLLECTION_NAME = "test-collection"
        
        mock_search.return_value = []
        
        main()
        
        captured = capsys.readouterr()
        assert "Search returned no results" in captured.out
