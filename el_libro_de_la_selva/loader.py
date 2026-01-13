import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Handles loading documents from text files.

    This class provides a simple interface for loading text documents
    using LangChain's TextLoader, which reads files and returns Document objects.
    """

    def load_from_file(self, file_path: str) -> list[Document]:
        """Load a document from a text file.

        Args:
            file_path: Path to the text file to load

        Returns:
            List of LangChain Document objects containing the file content

        Raises:
            FileNotFoundError: If the specified file does not exist
            UnicodeDecodeError: If the file encoding is not UTF-8
        """
        loader = TextLoader(file_path=file_path, encoding='utf-8')
        return loader.load()


class DocumentSplitter:
    """Splits documents into smaller, overlapping chunks.

    This class uses recursive character text splitting to break documents
    into manageable chunks while preserving context through overlap.

    Attributes:
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize the document splitter.

        Args:
            chunk_size: Maximum characters per text chunk
            chunk_overlap: Characters to overlap between consecutive chunks

        Raises:
            ValueError: If chunk_size is less than chunk_overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks with metadata.

        This method splits the input documents using recursive character splitting,
        then adds metadata to each chunk including:
        - level: Set to 1 indicating these are leaf nodes
        - id: Unique UUID for each chunk

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects with level and id metadata

        Note:
            The recursive splitter tries to maintain paragraph and sentence boundaries
            where possible while respecting the chunk_size constraint.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = text_splitter.split_documents(documents)
        for doc in split_docs:
            doc.metadata["level"] = 1
            doc.metadata["id"] = str(uuid.uuid4())
        return split_docs
