from langchain_qdrant import QdrantVectorStore
from qdrant_client import models


def collapsed_tree_retrieval(query: str, vector_store: QdrantVectorStore):
    """Retrieve documents without considering hierarchy levels.

    This function performs a flat similarity search across all documents
    in the vector store, regardless of their hierarchy level. It retrieves
    the top 10 most similar documents based on the query.

    Args:
        query: Search query string
        vector_store: QdrantVectorStore instance to search in

    Returns:
        List of Document objects representing the most similar documents

    Raises:
        Exception: If similarity search fails

    Example:
        >>> results = collapsed_tree_retrieval("Who is the protagonist?", vector_store)
        >>> print(len(results))
        10

    Note:
        This retrieval method is faster than tree_traversal_search but
        may return results from multiple hierarchy levels mixed together.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return retriever.invoke(query)


def tree_traversal_search(query: str, vector_store: QdrantVectorStore, max_level: int = 3):
    """Retrieve documents using hierarchical tree traversal.

    This function performs a two-stage retrieval that respects the document
    hierarchy:
    1. Finds the top 2 most similar summary documents at the specified max_level
    2. Retrieves the child documents of those summaries (up to 3 children total)

    This approach provides both high-level summaries and their corresponding
    detailed content, giving users context at different abstraction levels.

    Args:
        query: Search query string
        vector_store: QdrantVectorStore instance to search in
        max_level: Hierarchy level to start search from (default: 3)

    Returns:
        List of Document objects containing summary nodes and their child documents

    Raises:
        KeyError: If metadata is missing or malformed
        Exception: If similarity search fails

    Example:
        >>> results = tree_traversal_search("Who is Papa Wolf?", vector_store, max_level=3)
        >>> for doc in results:
        ...     print(f"Level {doc.metadata['level']}: {doc.page_content[:50]}")

    Note:
        This method is slower than collapsed_tree_retrieval but provides
        more structured results that follow the document hierarchy.
    """
    summary_results = vector_store.similarity_search(
        query,
        k=2,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.level",
                    match=models.MatchValue(value=max_level)
                )
            ]
        )
    )

    child_ids = []
    for doc in summary_results:
        print(f"doc metadata: {doc.metadata}")
        child_ids.extend(doc.metadata.get("children_ids", []))

    print(child_ids)

    detailed_results = vector_store.similarity_search(
        query,
        k=3,
        filter=models.Filter(must=[
            models.FieldCondition(key="metadata.id", match=models.MatchAny(any=child_ids))
        ])
    )

    return summary_results + detailed_results
