"""Unit tests for storage component."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from el_libro_de_la_selva.storage import QdrantStorage


class TestQdrantStorage:
    """Test cases for QdrantStorage class."""

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_valid_documents(self, mock_qdrant_class):
        """Test storing documents in Qdrant vector store."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [
            Document(page_content="First document", metadata={"id": "1"}),
            Document(page_content="Second document", metadata={"id": "2"}),
        ]
        
        mock_embeddings = MagicMock()
        collection_name = "test-collection"
        url = "http://localhost:6333"
        
        storage.store(docs, mock_embeddings, collection_name, url)
        
        mock_qdrant_class.from_documents.assert_called_once()
        call_args = mock_qdrant_class.from_documents.call_args
        assert call_args[1]["documents"] == docs
        assert call_args[1]["embedding"] == mock_embeddings
        assert call_args[1]["collection_name"] == collection_name
        assert call_args[1]["url"] == url

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_single_document(self, mock_qdrant_class):
        """Test storing a single document."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [Document(page_content="Single document", metadata={"id": "1"})]
        mock_embeddings = MagicMock()
        
        storage.store(docs, mock_embeddings, "test", "http://localhost:6333")
        
        mock_qdrant_class.from_documents.assert_called_once()

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_empty_document_list(self, mock_qdrant_class):
        """Test storing empty document list."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = []
        mock_embeddings = MagicMock()
        
        storage.store(docs, mock_embeddings, "test", "http://localhost:6333")
        
        mock_qdrant_class.from_documents.assert_called_once()

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_custom_url(self, mock_qdrant_class):
        """Test storing documents with custom URL."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [Document(page_content="Test", metadata={"id": "1"})]
        mock_embeddings = MagicMock()
        custom_url = "http://custom-qdrant:6333"
        
        storage.store(docs, mock_embeddings, "test", custom_url)
        
        call_args = mock_qdrant_class.from_documents.call_args
        assert call_args[1]["url"] == custom_url

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_custom_collection_name(self, mock_qdrant_class):
        """Test storing documents with custom collection name."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [Document(page_content="Test", metadata={"id": "1"})]
        mock_embeddings = MagicMock()
        custom_collection = "custom-collection"
        
        storage.store(docs, mock_embeddings, custom_collection, "http://localhost:6333")
        
        call_args = mock_qdrant_class.from_documents.call_args
        assert call_args[1]["collection_name"] == custom_collection

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_preserves_document_metadata(self, mock_qdrant_class):
        """Test that document metadata is preserved during storage."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [
            Document(
                page_content="Content",
                metadata={"id": "1", "level": 2, "children_ids": ["a", "b"]}
            ),
        ]
        mock_embeddings = MagicMock()
        
        storage.store(docs, mock_embeddings, "test", "http://localhost:6333")
        
        call_args = mock_qdrant_class.from_documents.call_args
        assert call_args[1]["documents"][0].metadata == docs[0].metadata

    @patch('el_libro_de_la_selva.storage.QdrantVectorStore')
    def test_store_with_unicode_content(self, mock_qdrant_class):
        """Test storing documents with Unicode content."""
        mock_qdrant_class.from_documents.return_value = MagicMock()
        
        storage = QdrantStorage()
        
        docs = [
            Document(page_content="Café 日本語 🦄", metadata={"id": "1"}),
        ]
        mock_embeddings = MagicMock()
        
        storage.store(docs, mock_embeddings, "test", "http://localhost:6333")
        
        mock_qdrant_class.from_documents.assert_called_once()
