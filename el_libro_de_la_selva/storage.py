from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


class QdrantStorage:
    """Manages vector storage operations using Qdrant.

    This class provides methods to store documents in a Qdrant vector database,
    creating embeddings and storing metadata for efficient similarity search.
    """

    def store(self, documents: list[Document], embeddings, collection_name: str, url: str):
        """Store documents in Qdrant vector database.

        This method creates a vector store from the provided documents, generating
        embeddings and storing them along with their metadata. If the collection
        does not exist, it will be automatically created.

        Args:
            documents: List of LangChain Document objects to store
            embeddings: LangChain embeddings object (e.g., OllamaEmbeddings)
            collection_name: Name of the Qdrant collection to store documents in
            url: URL of the Qdrant server (e.g., "http://localhost:6333")

        Raises:
            ConnectionError: If unable to connect to Qdrant server
            Exception: If document embedding or storage fails

        Note:
            This method appends to existing collections. To replace documents,
            delete the collection first or use a different collection name.
        """
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )
