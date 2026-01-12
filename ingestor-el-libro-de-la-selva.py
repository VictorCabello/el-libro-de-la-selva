from dataclasses import dataclass
from collections import defaultdict

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_qdrant import QdrantVectorStore
from sklearn.mixture import GaussianMixture


@dataclass
class Config:
    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: int = 0
    EMBEDDING_MODEL: str = "bge-m3"
    OLLAMA_URL: str = "http://localhost:11434"
    QDRANT_URL: str = "http://localhost:6333"
    INPUT_FILE: str = "El-libro-de-la-selva.txt"
    COLLECTION_NAME: str = "el-libro-de-la-selva"
    DEFAULT_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CLUSTERS: int = 5
    MAX_HIERARCHICAL_LAYERS: int = 3
    RANDOM_STATE: int = 42


class PromptTemplates:
    """Centralized prompt templates."""

    @staticmethod
    def summarization_template() -> str:
        return "Resume los siguientes fragmentos de texto en un solo párrafo cohesivo:\n\n{context}"

    @staticmethod
    def create_summarization_chain(llm) -> Runnable:
        template = PromptTemplates.summarization_template()
        prompt = ChatPromptTemplate.from_template(template=template)
        return prompt | llm


class DocumentLoader:
    """Load documents from files."""

    def load_from_file(self, file_path: str) -> list[Document]:
        loader = TextLoader(file_path=file_path, encoding='utf-8')
        return loader.load()


class DocumentSplitter:
    """Split documents into chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = text_splitter.split_documents(documents)
        for doc in split_docs:
            doc.metadata["level"] = 0
        return split_docs


class DocumentClusterer:
    """Cluster documents using embeddings."""

    def __init__(self, embeddings, max_clusters: int, random_state: int):
        self.embeddings = embeddings
        self.max_clusters = max_clusters
        self.random_state = random_state

    def cluster(self, documents: list[Document]) -> dict[int, list[Document]]:
        if len(documents) < 2:
            return {}

        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        n_clusters = min(len(vectors), self.max_clusters)

        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        labels = gmm.fit_predict(vectors)

        clusters = defaultdict(list)
        for doc, label in zip(documents, labels):
            clusters[label].append(doc)

        return dict(clusters)


class DocumentSummarizer:
    """Summarize document clusters."""

    def __init__(self, llm):
        self.llm = llm
        self.chain = PromptTemplates.create_summarization_chain(llm)

    def summarize_cluster(self, documents: list[Document], layer_number: int) -> Document:
        context = "\n\n".join([doc.page_content for doc in documents])
        summary_content = self.chain.invoke({"context": context}).content
        return Document(
            page_content=summary_content,
            metadata={"level": layer_number}
        )


class DocumentHierarchyBuilder:
    """Build hierarchical summaries of documents."""

    def __init__(self, clusterer: DocumentClusterer, summarizer: DocumentSummarizer, max_layers: int):
        self.clusterer = clusterer
        self.summarizer = summarizer
        self.max_layers = max_layers

    def _summarize_all_clusters(
        self, clusters: dict[int, list[Document]],
        layer_number: int
    ) -> list[Document]:
        summaries = []
        for cluster_id in clusters:
            summary = self.summarizer.summarize_cluster(
                clusters[cluster_id], layer_number=layer_number)
            summaries.append(summary)
        return summaries

    def _process_layer(self, documents: list[Document], layer_number: int) -> list[Document] | None:
        print(f"Procesando capa {layer_number} ...")
        clusters = self.clusterer.cluster(documents)
        if not clusters:
            return None
        return self._summarize_all_clusters(clusters, layer_number=layer_number)

    def build_hierarchy(self, documents: list[Document]) -> list[Document]:
        all_nodes = documents.copy()
        current_layer = documents

        for layer_num in range(self.max_layers):
            new_summaries = self._process_layer(current_layer, layer_num + 1)
            if new_summaries is None:
                break
            all_nodes.extend(new_summaries)
            current_layer = new_summaries

        return all_nodes


class QdrantStorage:
    """Store documents in Qdrant vector store."""

    def store(self, documents: list[Document], embeddings, collection_name: str, url: str):
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )


def main():
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
