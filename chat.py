from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import models

def collapsedTreeRetrival(query: str, vectorStore: QdrantVectorStore):
    # Standard retriever saerch - looks at all levels sumaries and chunks at once
    retriver = vectorStore.as_retriever(search_kwards={"k": 10})
    return retriver.invoke(query)
    

def tree_traversal_search(query, vector_store: QdrantVectorStore, max_level=3):
    # 1. Start with high-level summaries to get the broad context
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

    
    # 2. Extract child IDs from the best summaries to "drill down"
    child_ids = []
    for doc in summary_results:
        child_ids.extend(doc.metadata.get("children_ids", []))
    
    # 3. Search only within the children for specific details
    detailed_results = vector_store.similarity_search(
        query,
        k=3,
        filter=models.Filter(must=[
            models.FieldCondition(key="metadata.id", match=models.MatchAny(any=child_ids))
        ])
    )
    
    return summary_results + detailed_results

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="el-libro-de-la-selva",
)
query = "Quien es papa lobo?"

docs = collapsedTreeRetrival(query=query, vectorStore=qdrant)
if docs:
    print(f"Collapsed! Found {len(docs)} relevant chunks.")
    for doc in docs:
        print(f"level {doc.metadata['level']}\n{doc.page_content}")
else:
    print("Search returned no results. Ingestion may have failed.")


docs = tree_traversal_search(query=query, vector_store=qdrant)
if docs:
    print(f"tree trasversal! Found {len(docs)} relevant chunks.")
    for doc in docs:
        print(f"level {doc.metadata['level']}\n{doc.page_content}")
else:
    print("Search returned no results. Ingestion may have failed.")
