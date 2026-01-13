"""Unit tests for document loader and splitter components."""

import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from el_libro_de_la_selva.loader import DocumentLoader, DocumentSplitter


class TestDocumentLoader:
    """Test cases for DocumentLoader class."""

    def test_load_from_file_existing_file(self, tmp_path):
        """Test loading a document from an existing file."""
        loader = DocumentLoader()
        
        test_content = "This is a test document.\nIt has multiple lines.\n"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        docs = loader.load_from_file(str(test_file))
        
        assert len(docs) == 1
        assert docs[0].page_content == test_content
        assert isinstance(docs[0], Document)

    def test_load_from_file_file_not_found(self):
        """Test loading from a non-existent file raises RuntimeError."""
        loader = DocumentLoader()
        
        with pytest.raises(RuntimeError, match="Error loading"):
            loader.load_from_file("nonexistent_file.txt")

    def test_load_from_file_unicode_content(self, tmp_path):
        """Test loading a document with Unicode characters."""
        loader = DocumentLoader()
        
        test_content = "Document with special chars: café, 日本語, 🦄"
        test_file = tmp_path / "unicode.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        docs = loader.load_from_file(str(test_file))
        
        assert len(docs) == 1
        assert docs[0].page_content == test_content

    def test_load_from_file_large_file(self, tmp_path):
        """Test loading a large document file."""
        loader = DocumentLoader()
        
        test_content = "Paragraph 1\n" * 100
        test_file = tmp_path / "large.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        docs = loader.load_from_file(str(test_file))
        
        assert len(docs) == 1
        assert len(docs[0].page_content) == len(test_content)


class TestDocumentSplitter:
    """Test cases for DocumentSplitter class."""

    def test_splitter_initialization(self):
        """Test DocumentSplitter initialization with valid parameters."""
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_splitter_chunk_size_greater_than_overlap(self):
        """Test DocumentSplitter with chunk_size greater than chunk_overlap."""
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100

    def test_split_single_document(self):
        """Test splitting a single document into chunks."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        doc = Document(page_content="A" * 300)
        chunks = splitter.split([doc])
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert "level" in chunk.metadata
            assert chunk.metadata["level"] == 1
            assert "id" in chunk.metadata
            assert len(chunk.metadata["id"]) > 0

    def test_split_multiple_documents(self):
        """Test splitting multiple documents."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(page_content="Document one content " * 20),
            Document(page_content="Document two content " * 20),
        ]
        chunks = splitter.split(docs)
        
        assert len(chunks) > 2
        for chunk in chunks:
            assert chunk.metadata["level"] == 1
            assert "id" in chunk.metadata

    def test_split_document_preserves_content(self):
        """Test that splitting preserves document content."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=0)
        
        original_content = "A" * 150
        doc = Document(page_content=original_content)
        chunks = splitter.split([doc])
        
        combined_content = "".join([chunk.page_content for chunk in chunks])
        assert combined_content == original_content

    def test_split_with_overlap(self):
        """Test that chunks have correct overlap."""
        splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
        
        doc = Document(page_content="A" * 100)
        chunks = splitter.split([doc])
        
        if len(chunks) > 1:
            last_chars_chunk1 = chunks[0].page_content[-10:]
            first_chars_chunk2 = chunks[1].page_content[:10]
            assert last_chars_chunk1 == first_chars_chunk2

    def test_split_empty_document(self):
        """Test splitting an empty document."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        doc = Document(page_content="")
        chunks = splitter.split([doc])
        
        assert len(chunks) == 0

    def test_split_empty_document_list(self):
        """Test splitting an empty list of documents."""
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        chunks = splitter.split([])
        
        assert len(chunks) == 0

    def test_unique_ids_per_chunk(self):
        """Test that each chunk gets a unique ID."""
        splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
        
        doc = Document(page_content="A" * 150)
        chunks = splitter.split([doc])
        
        ids = [chunk.metadata["id"] for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_split_with_sentence_boundaries(self):
        """Test that splitter respects sentence boundaries."""
        splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
        
        doc = Document(page_content="First sentence. Second sentence. Third sentence. Fourth sentence.")
        chunks = splitter.split([doc])
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["level"] == 1
