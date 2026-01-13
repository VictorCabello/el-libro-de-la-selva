from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from el_libro_de_la_selva import Config, tree_traversal_search, collapsed_tree_retrieval


def main():
    """Main entry point for the chat CLI.

    This function demonstrates retrieval from the vector store using
    tree traversal search. It currently uses a hardcoded query for testing
    purposes.

    Note:
        This function is registered as the 'el-libro-de-la-selva-chat'
        command-line entry point in pyproject.toml.

        To make this interactive, modify the code to accept user input
        for queries instead of using the hardcoded query.
    """
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_URL
    )
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=Config.QDRANT_URL,
        collection_name=Config.COLLECTION_NAME,
    )
    query = "Quien es papa lobo?"

    print("\n\n")
    print("-" * 80)
    print("\n\n")

    docs = tree_traversal_search(query=query, vector_store=qdrant)
    if docs:
        print(f"tree traversal! Found {len(docs)} relevant chunks.")
        for doc in docs:
            print(f"level {doc.metadata['level']}\n{doc.page_content}")
    else:
        print("Search returned no results. Ingestion may have failed.")


if __name__ == "__main__":
    main()
