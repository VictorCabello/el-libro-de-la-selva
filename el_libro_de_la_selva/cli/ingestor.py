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


def main():
    """Main entry point for the document ingestion CLI.

    This function orchestrates the complete document ingestion pipeline:
    1. Loads configuration
    2. Initializes LLM and embedding models
    3. Creates pipeline components
    4. Loads and splits the source document
    5. Builds the hierarchical document structure
    6. Stores all documents in Qdrant

    Raises:
        FileNotFoundError: If the input file specified in Config does not exist
        ConnectionError: If unable to connect to Ollama or Qdrant
        Exception: If any step in the pipeline fails

    Note:
        This function is registered as the 'el-libro-de-la-selva-ingestor'
        command-line entry point in pyproject.toml.
    """
    config = Config()

    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_URL)

    loader = DocumentLoader()
    splitter = DocumentSplitter(config.DEFAULT_CHUNK_SIZE, config.CHUNK_OVERLAP)
    clusterer = DocumentClusterer(embeddings, config.MAX_CLUSTERS, config.RANDOM_STATE)
    summarizer = DocumentSummarizer(llm)
    builder = DocumentHierarchyBuilder(clusterer, summarizer, config.MAX_HIERARCHICAL_LAYERS)
    storage = QdrantStorage()

    docs = loader.load_from_file(config.INPUT_FILE)
    leaf_nodes = splitter.split(docs)
    all_nodes = builder.build_hierarchy(leaf_nodes)
    storage.store(all_nodes, embeddings, config.COLLECTION_NAME, config.QDRANT_URL)


if __name__ == "__main__":
    main()
