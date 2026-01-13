from collections import defaultdict
from langchain_core.documents import Document
from sklearn.mixture import GaussianMixture
from .prompts import PromptTemplates
import uuid


class DocumentClusterer:
    """Clusters documents based on their embedding similarity.

    This class uses Gaussian Mixture Models (GMM) to cluster documents
    in the embedding space, grouping semantically similar documents together.

    Attributes:
        embeddings: LangChain embeddings object for generating document vectors
        max_clusters: Maximum number of clusters to create
        random_state: Random seed for reproducible clustering results
    """

    def __init__(self, embeddings, max_clusters: int, random_state: int):
        """Initialize the document clusterer.

        Args:
            embeddings: LangChain embeddings object (e.g., OllamaEmbeddings)
            max_clusters: Maximum number of clusters to create
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If max_clusters is less than 1
        """
        self.embeddings = embeddings
        self.max_clusters = max_clusters
        self.random_state = random_state

    def cluster(self, documents: list[Document]) -> dict[int, list[Document]]:
        """Cluster documents based on their embeddings.

        This method converts documents to embeddings, applies Gaussian Mixture
        Model clustering, and returns the documents grouped by cluster ID.

        Args:
            documents: List of Document objects to cluster

        Returns:
            Dictionary mapping cluster IDs (int) to lists of Document objects.
            Returns empty dict if fewer than 2 documents are provided.

        Raises:
            Exception: If embedding generation or clustering fails

        Note:
            The actual number of clusters may be less than max_clusters if
            there are fewer documents than max_clusters.
        """
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
    """Creates summary documents for clustered content.

    This class uses an LLM to generate summaries for document clusters,
    preserving the relationship between summaries and their source documents.
    """

    def __init__(self, llm):
        """Initialize the document summarizer.

        Args:
            llm: LangChain LLM object for generating summaries (e.g., ChatOllama)

        Note:
            The LLM should be configured with temperature=0 for consistent results.
        """
        self.llm = llm
        self.chain = PromptTemplates.create_summarization_chain(llm)

    def summarize_cluster(self, documents: list[Document], layer_number: int) -> Document:
        """Summarize a cluster of documents into a single document.

        This method concatenates all document content, generates a summary using
        the LLM chain, and creates a new Document with the summary content and
        metadata tracking the relationship to source documents.

        Args:
            documents: List of Document objects in the cluster to summarize
            layer_number: Hierarchy level number for this summary (e.g., 2, 3, 4)

        Returns:
            Document object containing the summary with metadata:
                - level: The hierarchy layer number
                - children_ids: List of IDs from the source documents

        Raises:
            Exception: If LLM summarization fails
            KeyError: If source documents don't have IDs in metadata

        Note:
            The summarization prompt is in Spanish and requests a single cohesive paragraph.
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        summary_content = self.chain.invoke({"context": context}).content
        return Document(
            page_content=summary_content,
            metadata={
                "level": layer_number,
                "id": str(uuid.uuid4()),
                "children_ids": [doc.metadata.get("id") for doc in documents]
            },
        )


class DocumentHierarchyBuilder:
    """Builds a multi-layer document hierarchy through clustering and summarization.

    This class orchestrates the creation of a hierarchical tree structure where:
    - Layer 1: Original document chunks (leaf nodes)
    - Layer 2+: Summaries of clusters from the previous layer

    This hierarchical structure enables efficient retrieval at different levels of abstraction.
    """

    def __init__(self, clusterer: DocumentClusterer, summarizer: DocumentSummarizer, max_layers: int):
        """Initialize the hierarchy builder.

        Args:
            clusterer: DocumentClusterer instance for grouping documents
            summarizer: DocumentSummarizer instance for creating summaries
            max_layers: Maximum number of hierarchy layers to create

        Raises:
            ValueError: If max_layers is less than 1
        """
        self.clusterer = clusterer
        self.summarizer = summarizer
        self.max_layers = max_layers

    def _summarize_all_clusters(
        self, clusters: dict[int, list[Document]],
        layer_number: int
    ) -> list[Document]:
        """Summarize all clusters in a hierarchy layer.

        Args:
            clusters: Dictionary mapping cluster IDs to document lists
            layer_number: Hierarchy layer number for these summaries

        Returns:
            List of summary Document objects for all clusters
        """
        summaries = []
        for cluster_id in clusters:
            summary = self.summarizer.summarize_cluster(
                clusters[cluster_id], layer_number=layer_number)
            summaries.append(summary)
        return summaries

    def _process_layer(self, documents: list[Document], layer_number: int) -> list[Document] | None:
        """Process a single hierarchy layer: cluster and summarize.

        Args:
            documents: Documents at the current layer to process
            layer_number: The layer number being processed (1-indexed)

        Returns:
            List of summary documents for the next layer, or None if
            clustering is not possible (e.g., fewer than 2 documents)

        Note:
            This method prints progress to the console.
        """
        print(f"Procesando capa {layer_number} ...")
        clusters = self.clusterer.cluster(documents)
        if not clusters:
            return None
        return self._summarize_all_clusters(clusters, layer_number=layer_number)

    def build_hierarchy(self, documents: list[Document]) -> list[Document]:
        """Build the complete document hierarchy.

        This method creates a multi-layered hierarchy by iteratively:
        1. Clustering documents at the current layer
        2. Summarizing each cluster to create the next layer
        3. Repeating until max_layers is reached or no more clusters can be created

        Args:
            documents: Base documents (level 1) to start the hierarchy

        Returns:
            List of all Document objects in the hierarchy, including:
            - All original documents (level 1)
            - All summary documents from intermediate layers (level 2+)

        Note:
            The hierarchy may have fewer layers than max_layers if clustering
            is no longer possible (e.g., too few documents).
        """
        all_nodes = documents.copy()
        current_layer = documents

        for layer_num in range(self.max_layers):
            new_summaries = self._process_layer(current_layer, layer_num + 1)
            if new_summaries is None:
                break
            all_nodes.extend(new_summaries)
            current_layer = new_summaries

        return all_nodes
