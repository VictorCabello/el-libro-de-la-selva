# El Libro de la Selva Module Documentation

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [Config](#config)
  - [DocumentLoader](#documentloader)
  - [DocumentSplitter](#documentsplitter)
  - [DocumentClusterer](#documentclusterer)
  - [DocumentSummarizer](#documentsummarizer)
  - [DocumentHierarchyBuilder](#documenthierarchybuilder)
  - [QdrantStorage](#qdrantstorage)
  - [PromptTemplates](#prompttemplates)
  - [Retrieval Functions](#retrieval-functions)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage Examples](#usage-examples)
  - [Basic Ingestion Pipeline](#basic-ingestion-pipeline)
  - [Retrieval Operations](#retrieval-operations)
  - [Custom Configuration](#custom-configuration)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)
- [FAQ](#faq)

---

## Overview

The `el_libro_de_la_selva` module is a sophisticated document ingestion and retrieval system designed for processing large text collections. It implements a hierarchical clustering approach to create a multi-layered document tree, enabling efficient similarity-based retrieval at different levels of abstraction.

### Key Features

- **Hierarchical Document Processing**: Automatically clusters documents and creates summaries at multiple levels
- **Vector-based Storage**: Uses Qdrant vector database for efficient similarity search
- **Flexible Retrieval**: Supports both collapsed tree retrieval and hierarchical tree traversal
- **LangChain Integration**: Built on LangChain for seamless LLM and embedding operations
- **Ollama Support**: Runs locally with Ollama for embeddings and LLM operations

### Purpose

This module demonstrates how to build a scalable RAG (Retrieval-Augmented Generation) system by:
1. Ingesting and splitting documents into manageable chunks
2. Creating a hierarchical structure through clustering and summarization
3. Storing vectors in Qdrant for fast similarity search
4. Providing multiple retrieval strategies for different use cases

---

## Architecture

The system follows a pipeline architecture:

```
Text File → DocumentLoader → DocumentSplitter → [Leaf Nodes]
                                            ↓
                                    DocumentHierarchyBuilder
                                            ↓
                    [Clusterer → Summarizer] × N layers
                                            ↓
                                    QdrantStorage → Vector Database
                                            ↓
                            Retrieval Functions → Query Results
```

### Data Flow

1. **Loading**: Documents are loaded from text files using LangChain's TextLoader
2. **Splitting**: Recursive text splitting creates leaf-level chunks with metadata
3. **Hierarchy Building**: Documents are clustered and summarized across multiple layers
4. **Storage**: All nodes (leaf and summary) are stored as vectors in Qdrant
5. **Retrieval**: Queries can retrieve from any layer using different strategies

### Layer Structure

Each document in the hierarchy contains metadata:
- `level`: The hierarchy layer (1 = leaf, 2+ = summaries)
- `id`: Unique identifier for the document
- `children_ids`: List of child document IDs (for summary nodes)

---

## Components

### Config

**Location**: `config.py`

The `Config` dataclass provides centralized configuration management for the entire system.

**Attributes**:

```python
@dataclass
class Config:
    LLM_MODEL: str = "llama3.2"              # LLM model for summarization
    LLM_TEMPERATURE: int = 0                # Temperature for deterministic outputs
    EMBEDDING_MODEL: str = "bge-m3"          # Embedding model for vectors
    OLLAMA_URL: str = "http://localhost:11434"  # Ollama API endpoint
    QDRANT_URL: str = "http://localhost:6333"    # Qdrant endpoint
    INPUT_FILE: str = "El-libro-de-la-selva.txt" # Source text file
    COLLECTION_NAME: str = "el-libro-de-la-selva" # Qdrant collection name
    DEFAULT_CHUNK_SIZE: int = 1000           # Text chunk size
    CHUNK_OVERLAP: int = 200                 # Overlap between chunks
    MAX_CLUSTERS: int = 5                    # Maximum clusters per layer
    MAX_HIERARCHICAL_LAYERS: int = 3         # Maximum hierarchy depth
    RANDOM_STATE: int = 42                   # Random seed for reproducibility
```

**Usage**:

```python
from el_libro_de_la_selva import Config

config = Config()
# Override defaults
config.LLM_MODEL = "llama3.1"
config.MAX_CLUSTERS = 10
```

---

### DocumentLoader

**Location**: `loader.py`

Handles loading documents from text files.

**Methods**:

```python
class DocumentLoader:
    def load_from_file(self, file_path: str) -> list[Document]:
        """Load documents from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            List of LangChain Document objects
        """
```

**Example**:

```python
loader = DocumentLoader()
documents = loader.load_from_file("document.txt")
```

---

### DocumentSplitter

**Location**: `loader.py`

Splits documents into smaller, overlapping chunks using recursive character splitting.

**Methods**:

```python
class DocumentSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize the splitter.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents with level=1 and unique IDs
        """
```

**Example**:

```python
splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split(documents)
```

**Best Practices**:

- `chunk_size` should match your typical query length
- `chunk_overlap` (15-20%) ensures context continuity
- Adjust based on document type and LLM context window

---

### DocumentClusterer

**Location**: `clustering.py`

Clusters documents using Gaussian Mixture Models based on their embeddings.

**Methods**:

```python
class DocumentClusterer:
    def __init__(self, embeddings, max_clusters: int, random_state: int):
        """Initialize the clusterer.

        Args:
            embeddings: LangChain embeddings object
            max_clusters: Maximum number of clusters
            random_state: Random seed for reproducibility
        """

    def cluster(self, documents: list[Document]) -> dict[int, list[Document]]:
        """Cluster documents by embedding similarity.

        Args:
            documents: List of documents to cluster

        Returns:
            Dictionary mapping cluster IDs to document lists
        """
```

**Example**:

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3")
clusterer = DocumentClusterer(embeddings, max_clusters=5, random_state=42)
clusters = clusterer.cluster(documents)
```

**How It Works**:

1. Converts document text to embeddings
2. Uses Gaussian Mixture Model for clustering
3. Returns fewer than `max_clusters` if documents < max_clusters

---

### DocumentSummarizer

**Location**: `clustering.py`

Creates summary documents for clustered content using an LLM.

**Methods**:

```python
class DocumentSummarizer:
    def __init__(self, llm):
        """Initialize the summarizer.

        Args:
            llm: LangChain LLM object
        """

    def summarize_cluster(self, documents: list[Document], layer_number: int) -> Document:
        """Summarize a cluster of documents.

        Args:
            documents: List of documents in the cluster
            layer_number: Hierarchy level for the summary

        Returns:
            Summary document with children_ids metadata
        """
```

**Example**:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
summarizer = DocumentSummarizer(llm)
summary = summarizer.summarize_cluster(cluster_docs, layer_number=2)
```

**Summarization Prompt**:

The module uses a Spanish prompt: "Resume los siguientes fragmentos de texto en un solo párrafo cohesivo:\n\n{context}"

---

### DocumentHierarchyBuilder

**Location**: `clustering.py`

Orchestrates the creation of a multi-layer document hierarchy.

**Methods**:

```python
class DocumentHierarchyBuilder:
    def __init__(
        self,
        clusterer: DocumentClusterer,
        summarizer: DocumentSummarizer,
        max_layers: int
    ):
        """Initialize the hierarchy builder.

        Args:
            clusterer: DocumentClusterer instance
            summarizer: DocumentSummarizer instance
            max_layers: Maximum hierarchy depth
        """

    def build_hierarchy(self, documents: list[Document]) -> list[Document]:
        """Build the complete document hierarchy.

        Args:
            documents: Base documents (level 1)

        Returns:
            All nodes including leaf documents and all summaries
        """
```

**Example**:

```python
builder = DocumentHierarchyBuilder(clusterer, summarizer, max_layers=3)
all_nodes = builder.build_hierarchy(chunks)
```

**Process Flow**:

1. Starts with leaf documents (level 1)
2. Clusters current layer documents
3. Summarizes each cluster to create next layer
4. Repeats until max_layers or no more clusters possible
5. Returns all documents in the hierarchy

**Output Structure**:

- Layer 1: Original chunks (leaf nodes)
- Layer 2: Summaries of layer 1 clusters
- Layer 3: Summaries of layer 2 clusters
- etc.

---

### QdrantStorage

**Location**: `storage.py`

Manages vector storage operations using Qdrant.

**Methods**:

```python
class QdrantStorage:
    def store(
        self,
        documents: list[Document],
        embeddings,
        collection_name: str,
        url: str
    ):
        """Store documents in Qdrant.

        Args:
            documents: List of documents to store
            embeddings: LangChain embeddings object
            collection_name: Qdrant collection name
            url: Qdrant server URL
        """
```

**Example**:

```python
storage = QdrantStorage()
storage.store(
    documents=all_nodes,
    embeddings=embeddings,
    collection_name="my-docs",
    url="http://localhost:6333"
)
```

**Notes**:

- Automatically creates collection if it doesn't exist
- Stores metadata (level, id, children_ids) with each vector
- Uses QdrantVectorStore.from_documents for easy setup

---

### PromptTemplates

**Location**: `prompts.py`

Provides reusable prompt templates for LLM operations.

**Methods**:

```python
class PromptTemplates:
    @staticmethod
    def summarization_template() -> str:
        """Return the summarization prompt template."""

    @staticmethod
    def create_summarization_chain(llm) -> Runnable:
        """Create a summarization chain.

        Args:
            llm: LangChain LLM object

        Returns:
            Runnable chain for summarization
        """
```

**Example**:

```python
llm = ChatOllama(model="llama3.2")
chain = PromptTemplates.create_summarization_chain(llm)
result = chain.invoke({"context": document_text})
```

---

### Retrieval Functions

**Location**: `retrieval.py`

Provides two strategies for retrieving documents from the vector store.

#### collapsed_tree_retrieval

```python
def collapsed_tree_retrieval(
    query: str,
    vector_store: QdrantVectorStore
):
    """Retrieve documents without considering hierarchy.

    Args:
        query: Search query
        vector_store: QdrantVectorStore instance

    Returns:
        Top k most similar documents from all layers
    """
```

**Use Cases**:

- Quick searches where hierarchy doesn't matter
- When you want the most relevant content regardless of level
- Simpler retrieval with no hierarchy constraints

**Example**:

```python
results = collapsed_tree_retrieval(query="Who is Papa Wolf?", vector_store=vs)
```

#### tree_traversal_search

```python
def tree_traversal_search(
    query: str,
    vector_store: QdrantVectorStore,
    max_level: int = 3
):
    """Retrieve documents using hierarchical tree traversal.

    Args:
        query: Search query
        vector_store: QdrantVectorStore instance
        max_level: Starting level for search (default: 3)

    Returns:
        Summary nodes from max_level + their child documents
    """
```

**Use Cases**:

- Structured search following the document hierarchy
- When you want both summaries and their details
- More sophisticated retrieval with context preservation

**Algorithm**:

1. Find top 2 most similar summaries at `max_level`
2. Extract all `children_ids` from these summaries
3. Retrieve top 3 child documents matching those IDs
4. Return both summaries and detailed documents

**Example**:

```python
results = tree_traversal_search(
    query="Who is Papa Wolf?",
    vector_store=vs,
    max_level=3
)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Ollama running on http://localhost:11434 with bge-m3 model
- Qdrant running on http://localhost:6333

### Install Dependencies

```bash
pip install -e .
```

Or install directly:

```bash
pip install langchain-community \
            langchain-text-splitters \
            langchain-ollama \
            langchain-qdrant \
            numpy \
            scikit-learn
```

---

## Prerequisites

### Ollama Setup

1. Install Ollama: https://ollama.ai/download
2. Pull required models:

```bash
ollama pull llama3.2
ollama pull bge-m3
```

3. Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

### Qdrant Setup

#### Option 1: Docker (Recommended)

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

#### Option 2: Self-Hosted AI Starter Kit

Use the n8n self-hosted AI starter kit for easy setup:
https://github.com/n8n-io/self-hosted-ai-starter-kit

#### Option 3: Qdrant Cloud

- Sign up at https://cloud.qdrant.io
- Get your API URL and key
- Update Config.QDRANT_URL accordingly

---

## Usage Examples

### Basic Ingestion Pipeline

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from el_libro_de_la_selva import (
    Config,
    DocumentLoader,
    DocumentSplitter,
    DocumentClusterer,
    DocumentSummarizer,
    DocumentHierarchyBuilder,
    QdrantStorage,
)

# Initialize components
config = Config()
llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_URL)

# Create pipeline components
loader = DocumentLoader()
splitter = DocumentSplitter(config.DEFAULT_CHUNK_SIZE, config.CHUNK_OVERLAP)
clusterer = DocumentClusterer(embeddings, config.MAX_CLUSTERS, config.RANDOM_STATE)
summarizer = DocumentSummarizer(llm)
builder = DocumentHierarchyBuilder(clusterer, summarizer, config.MAX_HIERARCHICAL_LAYERS)
storage = QdrantStorage()

# Execute pipeline
docs = loader.load_from_file(config.INPUT_FILE)
leaf_nodes = splitter.split(docs)
all_nodes = builder.build_hierarchy(leaf_nodes)
storage.store(all_nodes, embeddings, config.COLLECTION_NAME, config.QDRANT_URL)

print(f"Processed {len(all_nodes)} total nodes across {config.MAX_HIERARCHICAL_LAYERS} layers")
```

### Retrieval Operations

```python
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from el_libro_de_la_selva import Config, tree_traversal_search, collapsed_tree_retrieval

# Connect to vector store
embeddings = OllamaEmbeddings(
    model=Config.EMBEDDING_MODEL,
    base_url=Config.OLLAMA_URL
)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=Config.QDRANT_URL,
    collection_name=Config.COLLECTION_NAME,
)

# Method 1: Collapsed tree retrieval
query = "Who is Papa Wolf?"
results = collapsed_tree_retrieval(query, vector_store)
print(f"Found {len(results)} results (collapsed tree)")
for doc in results:
    print(f"Level {doc.metadata['level']}: {doc.page_content[:100]}...")

# Method 2: Tree traversal search
results = tree_traversal_search(query, vector_store, max_level=3)
print(f"Found {len(results)} results (tree traversal)")
for doc in results:
    print(f"Level {doc.metadata['level']}: {doc.page_content[:100]}...")
```

### Custom Configuration

```python
from el_libro_de_la_selva import Config

# Create custom config
custom_config = Config(
    LLM_MODEL="llama3.1",
    LLM_TEMPERATURE=0,
    EMBEDDING_MODEL="bge-m3",
    OLLAMA_URL="http://localhost:11434",
    QDRANT_URL="http://localhost:6333",
    INPUT_FILE="my_document.txt",
    COLLECTION_NAME="my-collection",
    DEFAULT_CHUNK_SIZE=500,
    CHUNK_OVERLAP=100,
    MAX_CLUSTERS=10,
    MAX_HIERARCHICAL_LAYERS=4,
    RANDOM_STATE=42,
)

# Use custom config in your pipeline
```

---

## Common Pitfalls

### 1. Forgetting to Start Services

**Problem**: Connection refused errors to Ollama or Qdrant.

**Solution**: Always verify services are running before ingestion:

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/
```

### 2. Wrong Chunk Size

**Problem**: Chunks too small (loss of context) or too large (inefficient retrieval).

**Solution**:
- Start with 800-1200 characters for most text
- Adjust overlap to 15-20% of chunk size
- Test retrieval quality and adjust accordingly

### 3. Too Many Clusters

**Problem**: `MAX_CLUSTERS` set too high creates many small clusters, leading to:
- Slow summarization
- Fragmented summaries
- Poor hierarchy quality

**Solution**:
- Use 3-5 clusters for most use cases
- Increase only for very large document sets (1000+ chunks)

### 4. Not Enough Documents for Clustering

**Problem**: Clustering fails silently with < 2 documents.

**Solution**: The code handles this by returning empty dict, but be aware of the limitation. Use meaningful input files.

### 5. Temperature Too High

**Problem**: LLM temperature > 0 causes inconsistent summaries.

**Solution**: Always use `temperature=0` for summarization to ensure deterministic outputs.

### 6. Collection Already Exists

**Problem**: Running ingestion twice adds duplicate documents.

**Solution**: Either:
- Delete the collection before re-ingesting
- Use a different collection name
- Check if collection exists first

### 7. Memory Issues with Large Documents

**Problem**: Processing large text files (>100MB) causes memory errors.

**Solution**:
- Process in batches
- Reduce chunk size
- Use streaming where possible
- Increase system memory or use cloud resources

### 8. Wrong Embedding Model

**Problem**: Using LLM instead of embedding model for vectors.

**Solution**: Always use `OllamaEmbeddings` with `bge-m3` or similar embedding model, not `ChatOllama`.

---

## Best Practices

### Performance Optimization

1. **Batch Processing**: Process large documents in batches to avoid memory issues
2. **Parallel Embedding**: Use batch embedding when possible (LangChain supports this)
3. **Adjust k Values**: Tune retrieval `k` values based on your use case:
   - Small k (2-3): Focused results
   - Medium k (5-10): Balanced coverage
   - Large k (20+): Comprehensive search

### Quality Improvement

1. **Data Cleaning**: Clean input text before ingestion
2. **Custom Prompts**: Modify `PromptTemplates` for specific domains
3. **Layer Management**: Start with 2-3 layers, adjust based on document count
4. **Embedding Selection**: Test different embedding models for your domain

### Monitoring and Debugging

1. **Log Progress**: The hierarchy builder already logs layer progress
2. **Inspect Metadata**: Always check document metadata after processing
3. **Test Retrieval**: Verify retrieval quality before production use
4. **Monitor Qdrant**: Use Qdrant dashboard to inspect collections

### Security Considerations

1. **Secure Endpoints**: Use authentication for Qdrant in production
2. **Input Validation**: Validate file paths and content before processing
3. **Rate Limiting**: Implement rate limiting for production APIs
4. **Environment Variables**: Store sensitive URLs in environment variables

---

## FAQ

### Q1: What is the difference between the two retrieval methods?

**A**: `collapsed_tree_retrieval` searches across all layers equally, while `tree_traversal_search` follows the hierarchy:
- Use `collapsed_tree_retrieval` for simple, fast searches
- Use `tree_traversal_search` when you need both summaries and details in a structured way

### Q2: How many layers should I use?

**A**: Depends on your document count:
- 10-50 documents: 2 layers
- 50-200 documents: 3 layers
- 200-1000 documents: 4 layers
- 1000+ documents: 5+ layers

Start with 3 layers and adjust based on results.

### Q3: Can I use this with PDFs or other formats?

**A**: Yes! Replace `DocumentLoader` with any LangChain loader:
- PDF: `PyPDFLoader`
- Word: `Docx2txtLoader`
- Web: `WebBaseLoader`

### Q4: How do I handle Spanish text in prompts?

**A**: The module already uses Spanish prompts in `PromptTemplates`. For other languages:
1. Modify the prompt template in `prompts.py`
2. Use an LLM trained on your target language
3. Consider using translation for better quality

### Q5: What if I don't want to use Ollama?

**A**: You can replace Ollama with any LangChain-compatible service:
- OpenAI: `OpenAIEmbeddings`, `ChatOpenAI`
- HuggingFace: `HuggingFaceEmbeddings`
- Cohere: `CohereEmbeddings`, `ChatCohere`

Just update the embedding and LLM initialization.

### Q6: How do I update documents in Qdrant?

**A**: Qdrant doesn't support direct updates. To refresh:
```python
# Delete old collection
from qdrant_client import QdrantClient

client = QdrantClient(url=Config.QDRANT_URL)
client.delete_collection(Config.COLLECTION_NAME)

# Re-run ingestion
# ... (your ingestion code)
```

### Q7: Can I search across multiple collections?

**A**: Yes, create multiple vector stores and merge results:
```python
vs1 = QdrantVectorStore.from_existing_collection(embeddings, url, "collection1")
vs2 = QdrantVectorStore.from_existing_collection(embeddings, url, "collection2")

results1 = vs1.similarity_search(query, k=5)
results2 = vs2.similarity_search(query, k=5)

# Merge and re-rank
all_results = results1 + results2
# ... re-ranking logic
```

### Q8: How do I handle documents larger than context window?

**A**: The splitter handles this automatically by:
- Using recursive character splitting
- Respecting the chunk_size parameter
- Ensuring no chunk exceeds the context window

### Q9: What happens if clustering fails?

**A**: The `DocumentHierarchyBuilder` handles this gracefully:
- If < 2 documents, clustering returns empty dict
- The builder stops and returns all nodes created so far
- You'll get a partial hierarchy (valid but incomplete)

### Q10: Can I customize the clustering algorithm?

**A**: Yes! Replace `GaussianMixture` in `clustering.py`:
```python
from sklearn.cluster import KMeans

class DocumentClusterer:
    def cluster(self, documents: list[Document]) -> dict[int, list[Document]]:
        # ...
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(vectors)
        # ...
```

### Q11: How do I optimize for faster ingestion?

**A**: Several strategies:
1. Reduce `MAX_HIERARCHICAL_LAYERS`
2. Increase `DEFAULT_CHUNK_SIZE` (fewer chunks)
3. Reduce `MAX_CLUSTERS` (fewer summaries)
4. Use faster embedding models
5. Process during off-peak hours

### Q12: Can I use this for real-time document processing?

**A**: The current implementation is batch-oriented. For real-time:
1. Store new documents in a separate collection
2. Periodically merge and re-cluster
3. Consider incremental clustering approaches
4. Use message queues (RabbitMQ, Kafka) for processing

### Q13: How do I handle documents in multiple languages?

**A**: Options:
1. Use multilingual embedding models (e.g., `bge-m3` supports multiple languages)
2. Translate documents to a single language before processing
3. Create separate collections per language
4. Use language-aware clustering

### Q14: What if I get "model not found" errors?

**A**: Pull the required models:
```bash
ollama pull llama3.2  # LLM for summarization
ollama pull bge-m3    # Embeddings for vectors
```

### Q15: Can I deploy this to production?

**A**: Yes, but consider:
1. Use Qdrant Cloud instead of local instance
2. Implement authentication
3. Add monitoring and logging
4. Use environment variables for configuration
5. Implement rate limiting
6. Add error handling and retries
7. Consider containerization (Docker)
8. Set up CI/CD pipeline

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- [Vector Database Best Practices](https://www.pinecone.io/learn/vector-database/)

---

## Contributing

This is a learning project. Feel free to fork, modify, and experiment with the codebase!

## License

Check the project repository for license information.
