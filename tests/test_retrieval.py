"""Unit tests for retrieval functions."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from el_libro_de_la_selva.retrieval import collapsed_tree_retrieval, tree_traversal_search


class TestCollapsedTreeRetrieval:
    """Test cases for collapsed_tree_retrieval function."""

    def test_retrieval_with_mock_vector_store(self, mock_vector_store):
        """Test collapsed tree retrieval with mocked vector store."""
        query = "test query"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content=f"Result {i}", metadata={"level": i % 3 + 1})
            for i in range(10)
        ]
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        results = collapsed_tree_retrieval(query, mock_vector_store)
        
        assert len(results) == 10
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
        mock_retriever.invoke.assert_called_once_with(query)

    def test_retrieval_returns_documents(self, mock_vector_store):
        """Test that retrieval returns Document objects."""
        query = "test query"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Result", metadata={"level": 1})
        ]
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        results = collapsed_tree_retrieval(query, mock_vector_store)
        
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieval_with_empty_results(self, mock_vector_store):
        """Test retrieval when no results are found."""
        query = "nonexistent query"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        results = collapsed_tree_retrieval(query, mock_vector_store)
        
        assert len(results) == 0

    def test_retrieval_uses_k_parameter_correctly(self, mock_vector_store):
        """Test that retrieval uses k=10 parameter."""
        query = "test query"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        collapsed_tree_retrieval(query, mock_vector_store)
        
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


class TestTreeTraversalSearch:
    """Test cases for tree_traversal_search function."""

    def test_search_with_mock_vector_store(self, mock_vector_store):
        """Test tree traversal search with mocked vector store."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [
            [
                Document(
                    page_content="Summary 1",
                    metadata={"level": 3, "children_ids": ["child1", "child2"]}
                ),
                Document(
                    page_content="Summary 2",
                    metadata={"level": 3, "children_ids": ["child3"]}
                ),
            ],
            [
                Document(page_content="Child 1", metadata={"level": 1, "id": "child1"}),
                Document(page_content="Child 2", metadata={"level": 1, "id": "child2"}),
                Document(page_content="Child 3", metadata={"level": 1, "id": "child3"}),
            ],
        ]
        
        results = tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        assert len(results) == 5

    def test_search_with_default_max_level(self, mock_vector_store):
        """Test tree traversal search with default max_level=3."""
        query = "test query"
        
        mock_vector_store.similarity_search.side_effect = [[], []]
        
        results = tree_traversal_search(query, mock_vector_store)
        
        assert results == []
        mock_vector_store.similarity_search.assert_called()

    def test_search_with_custom_max_level(self, mock_vector_store):
        """Test tree traversal search with custom max_level."""
        query = "test query"
        max_level = 2
        
        mock_vector_store.similarity_search.side_effect = [[], []]
        
        tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        first_call_args = mock_vector_store.similarity_search.call_args_list[0]
        assert first_call_args[1]["k"] == 2

    def test_search_with_no_summary_results(self, mock_vector_store):
        """Test search when no summaries are found at max_level."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [[], []]
        
        results = tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        assert len(results) == 0

    def test_search_with_no_children(self, mock_vector_store):
        """Test search when summaries have no children_ids."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [
            [
                Document(
                    page_content="Summary",
                    metadata={"level": 3, "children_ids": []}
                ),
            ],
            [],
        ]
        
        results = tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        assert len(results) == 1

    def test_search_with_missing_children_ids(self, mock_vector_store):
        """Test search when children_ids key is missing from metadata."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [
            [
                Document(
                    page_content="Summary",
                    metadata={"level": 3}
                ),
            ],
            [],
        ]
        
        results = tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        assert len(results) == 1

    def test_search_retrieves_two_summaries(self, mock_vector_store):
        """Test that search retrieves exactly 2 summary documents."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [[], []]
        
        tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        first_call_args = mock_vector_store.similarity_search.call_args_list[0]
        assert first_call_args[1]["k"] == 2

    def test_search_retrieves_up_to_three_children(self, mock_vector_store):
        """Test that search retrieves up to 3 child documents."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [
            [
                Document(
                    page_content="Summary",
                    metadata={"level": 3, "children_ids": ["c1", "c2", "c3", "c4", "c5"]}
                ),
            ],
            [
                Document(page_content="Child 1", metadata={"id": "c1"}),
            ],
        ]
        
        tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        second_call_args = mock_vector_store.similarity_search.call_args_list[1]
        assert second_call_args[1]["k"] == 3

    def test_search_returns_summaries_and_children(self, mock_vector_store):
        """Test that search returns both summaries and their children."""
        query = "test query"
        max_level = 3
        
        mock_vector_store.similarity_search.side_effect = [
            [
                Document(
                    page_content="Summary",
                    metadata={"level": 3, "children_ids": ["child1"]}
                ),
            ],
            [
                Document(page_content="Child", metadata={"id": "child1", "level": 1}),
            ],
        ]
        
        results = tree_traversal_search(query, mock_vector_store, max_level=max_level)
        
        assert len(results) == 2
        assert any(doc.metadata["level"] == 3 for doc in results)
        assert any(doc.metadata.get("level") == 1 for doc in results)


@pytest.fixture
def mock_vector_store():
    """Create a mock QdrantVectorStore for testing."""
    mock = MagicMock()
    return mock
