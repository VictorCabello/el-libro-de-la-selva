"""Unit tests for document clustering, summarization, and hierarchy building."""

import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock
from langchain_core.documents import Document
from el_libro_de_la_selva.clustering import (
    DocumentClusterer,
    DocumentSummarizer,
    DocumentHierarchyBuilder,
)


class TestDocumentClusterer:
    """Test cases for DocumentClusterer class."""

    def test_clusterer_initialization(self, mock_embeddings):
        """Test DocumentClusterer initialization with valid parameters."""
        clusterer = DocumentClusterer(
            embeddings=mock_embeddings,
            max_clusters=5,
            random_state=42
        )
        
        assert clusterer.embeddings == mock_embeddings
        assert clusterer.max_clusters == 5
        assert clusterer.random_state == 42

    def test_cluster_with_fewer_documents_than_clusters(self, mock_embeddings):
        """Test clustering when fewer documents than max_clusters."""
        mock_embeddings.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
        
        clusterer = DocumentClusterer(
            embeddings=mock_embeddings,
            max_clusters=10,
            random_state=42
        )
        
        docs = [
            Document(page_content="First document"),
            Document(page_content="Second document"),
        ]
        
        clusters = clusterer.cluster(docs)
        
        assert isinstance(clusters, dict)
        assert len(clusters) <= len(docs)

    def test_cluster_with_single_document(self, mock_embeddings):
        """Test clustering with a single document returns empty dict."""
        clusterer = DocumentClusterer(
            embeddings=mock_embeddings,
            max_clusters=5,
            random_state=42
        )
        
        docs = [Document(page_content="Single document")]
        
        clusters = clusterer.cluster(docs)
        
        assert clusters == {}

    def test_cluster_with_empty_list(self, mock_embeddings):
        """Test clustering with empty document list returns empty dict."""
        clusterer = DocumentClusterer(
            embeddings=mock_embeddings,
            max_clusters=5,
            random_state=42
        )
        
        clusters = clusterer.cluster([])
        
        assert clusters == {}

    def test_cluster_creates_multiple_clusters(self, mock_embeddings):
        """Test that clustering can create multiple clusters."""
        mock_embeddings.embed_documents.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        
        clusterer = DocumentClusterer(
            embeddings=mock_embeddings,
            max_clusters=3,
            random_state=42
        )
        
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]
        
        clusters = clusterer.cluster(docs)
        
        assert len(clusters) >= 1
        for cluster_id, cluster_docs in clusters.items():
            assert isinstance(cluster_id, (int, type(cluster_id)))
            assert isinstance(cluster_docs, list)
            assert all(isinstance(doc, Document) for doc in cluster_docs)

    def test_cluster_with_reproducible_results(self, mock_embeddings):
        """Test that clustering is reproducible with same random state."""
        mock_embeddings.embed_documents.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]
        
        clusterer1 = DocumentClusterer(mock_embeddings, 2, 42)
        clusters1 = clusterer1.cluster(docs)
        
        clusterer2 = DocumentClusterer(mock_embeddings, 2, 42)
        clusters2 = clusterer2.cluster(docs)
        
        assert len(clusters1) == len(clusters2)


class TestDocumentSummarizer:
    """Test cases for DocumentSummarizer class."""

    @patch('el_libro_de_la_selva.clustering.PromptTemplates')
    def test_summarizer_initialization(self, mock_prompts, mock_llm):
        """Test DocumentSummarizer initialization."""
        mock_chain = MagicMock()
        mock_prompts.create_summarization_chain.return_value = mock_chain
        
        summarizer = DocumentSummarizer(mock_llm)
        
        assert summarizer.llm == mock_llm
        assert summarizer.chain is not None
        mock_prompts.create_summarization_chain.assert_called_once_with(mock_llm)

    @patch('el_libro_de_la_selva.clustering.PromptTemplates')
    def test_summarize_cluster_with_multiple_documents(self, mock_prompts, mock_llm):
        """Test summarizing a cluster of multiple documents."""
        mock_chain = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Test summary"
        mock_chain.invoke.return_value = mock_result
        mock_prompts.create_summarization_chain.return_value = mock_chain
        
        summarizer = DocumentSummarizer(mock_llm)
        
        docs = [
            Document(page_content="First document content", metadata={"id": "id1"}),
            Document(page_content="Second document content", metadata={"id": "id2"}),
        ]
        
        summary = summarizer.summarize_cluster(docs, layer_number=2)
        
        assert isinstance(summary, Document)
        assert summary.metadata["level"] == 2
        assert "children_ids" in summary.metadata
        assert len(summary.metadata["children_ids"]) == 2

    @patch('el_libro_de_la_selva.clustering.PromptTemplates')
    def test_summarize_cluster_with_single_document(self, mock_prompts, mock_llm):
        """Test summarizing a cluster with a single document."""
        mock_chain = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Single doc summary"
        mock_chain.invoke.return_value = mock_result
        mock_prompts.create_summarization_chain.return_value = mock_chain
        
        summarizer = DocumentSummarizer(mock_llm)
        
        docs = [
            Document(page_content="Single document", metadata={"id": "id1"}),
        ]
        
        summary = summarizer.summarize_cluster(docs, layer_number=2)
        
        assert isinstance(summary, Document)
        assert summary.metadata["level"] == 2
        assert summary.metadata["children_ids"] == ["id1"]

    @patch('el_libro_de_la_selva.clustering.PromptTemplates')
    def test_summarize_cluster_sets_correct_metadata(self, mock_prompts, mock_llm):
        """Test that summarization sets correct metadata."""
        mock_chain = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Summary"
        mock_chain.invoke.return_value = mock_result
        mock_prompts.create_summarization_chain.return_value = mock_chain
        
        summarizer = DocumentSummarizer(mock_llm)
        
        docs = [
            Document(page_content="Doc 1", metadata={"id": "id1"}),
            Document(page_content="Doc 2", metadata={"id": "id2"}),
        ]
        
        summary = summarizer.summarize_cluster(docs, layer_number=3)
        
        assert summary.metadata["level"] == 3
        assert summary.metadata["children_ids"] == ["id1", "id2"]
        assert "id2" not in summary.metadata

    @patch('el_libro_de_la_selva.clustering.PromptTemplates')
    def test_summarize_cluster_different_layers(self, mock_prompts, mock_llm):
        """Test summarizing clusters for different hierarchy layers."""
        mock_chain = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Layer summary"
        mock_chain.invoke.return_value = mock_result
        mock_prompts.create_summarization_chain.return_value = mock_chain
        
        summarizer = DocumentSummarizer(mock_llm)
        
        docs = [Document(page_content="Content", metadata={"id": "id1"})]
        
        for layer in [2, 3, 4, 5]:
            summary = summarizer.summarize_cluster(docs, layer_number=layer)
            assert summary.metadata["level"] == layer


class TestDocumentHierarchyBuilder:
    """Test cases for DocumentHierarchyBuilder class."""

    def test_hierarchy_builder_initialization(self, mock_clusterer, mock_summarizer):
        """Test DocumentHierarchyBuilder initialization."""
        builder = DocumentHierarchyBuilder(
            clusterer=mock_clusterer,
            summarizer=mock_summarizer,
            max_layers=3
        )
        
        assert builder.clusterer == mock_clusterer
        assert builder.summarizer == mock_summarizer
        assert builder.max_layers == 3

    def test_summarize_all_clusters(self, mock_clusterer, mock_summarizer):
        """Test summarizing all clusters in a layer."""
        mock_summarizer.summarize_cluster.side_effect = [
            Document(page_content="Summary 1", metadata={"level": 2, "children_ids": ["id1"]}),
            Document(page_content="Summary 2", metadata={"level": 2, "children_ids": ["id2"]}),
        ]
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 3)
        
        clusters = {
            0: [Document(page_content="Doc 1", metadata={"id": "id1"})],
            1: [Document(page_content="Doc 2", metadata={"id": "id2"})],
        }
        
        summaries = builder._summarize_all_clusters(clusters, layer_number=2)
        
        assert len(summaries) == 2
        assert all(isinstance(s, Document) for s in summaries)

    def test_process_layer_returns_summaries(self, mock_clusterer, mock_summarizer):
        """Test that _process_layer returns summaries for a layer."""
        mock_clusterer.cluster.return_value = {
            0: [Document(page_content="Doc 1", metadata={"id": "id1"})]
        }
        mock_summarizer.summarize_cluster.return_value = Document(
            page_content="Summary",
            metadata={"level": 2, "children_ids": ["id1"]}
        )
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 3)
        
        docs = [Document(page_content="Doc 1", metadata={"id": "id1"})]
        result = builder._process_layer(docs, 1)
        
        assert result is not None
        assert len(result) == 1
        assert result[0].metadata["level"] == 2

    def test_process_layer_returns_none_on_no_clusters(self, mock_clusterer, mock_summarizer):
        """Test that _process_layer returns None when clustering fails."""
        mock_clusterer.cluster.return_value = {}
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 3)
        
        docs = [Document(page_content="Single doc", metadata={"id": "id1"})]
        result = builder._process_layer(docs, 1)
        
        assert result is None

    def test_build_hierarchy_creates_multiple_layers(self, mock_clusterer, mock_summarizer):
        """Test that build_hierarchy creates multiple hierarchy layers."""
        mock_clusterer.cluster.return_value = {
            0: [Document(page_content="Doc", metadata={"id": "id1"})]
        }
        mock_summarizer.summarize_cluster.return_value = Document(
            page_content="Summary",
            metadata={"level": 2, "children_ids": ["id1"]}
        )
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 2)
        
        docs = [
            Document(page_content="Doc 1", metadata={"id": "id1"}),
            Document(page_content="Doc 2", metadata={"id": "id2"}),
        ]
        
        hierarchy = builder.build_hierarchy(docs)
        
        assert len(hierarchy) > len(docs)

    def test_build_hierarchy_with_single_document(self, mock_clusterer, mock_summarizer):
        """Test building hierarchy with a single document."""
        mock_clusterer.cluster.return_value = {}
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 3)
        
        docs = [Document(page_content="Single", metadata={"id": "id1"})]
        hierarchy = builder.build_hierarchy(docs)
        
        assert len(hierarchy) == 1

    def test_build_hierarchy_stops_at_max_layers(self, mock_clusterer, mock_summarizer):
        """Test that build_hierarchy respects max_layers parameter."""
        mock_clusterer.cluster.return_value = {
            0: [Document(page_content="Doc", metadata={"id": "id1"})]
        }
        layer_num = 2
        mock_summarizer.summarize_cluster.return_value = Document(
            page_content="Summary",
            metadata={"level": layer_num, "children_ids": ["id1"]}
        )
        
        builder = DocumentHierarchyBuilder(mock_clusterer, mock_summarizer, 2)
        
        docs = [
            Document(page_content="Doc 1", metadata={"id": "id1"}),
            Document(page_content="Doc 2", metadata={"id": "id2"}),
        ]
        
        hierarchy = builder.build_hierarchy(docs)
        
        max_level = max(doc.metadata.get("level", 1) for doc in hierarchy)
        assert max_level <= 2


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    mock = MagicMock()
    mock.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return mock


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test summary")
    return mock


@pytest.fixture
def mock_clusterer(mock_embeddings):
    """Create mock clusterer for testing."""
    mock = MagicMock()
    mock.embeddings = mock_embeddings
    mock.max_clusters = 5
    mock.random_state = 42
    return mock


@pytest.fixture
def mock_summarizer(mock_llm):
    """Create mock summarizer for testing."""
    mock = MagicMock()
    mock.llm = mock_llm
    mock.summarize_cluster.return_value = Document(
        page_content="Summary",
        metadata={"level": 2, "children_ids": ["id1"]}
    )
    return mock
