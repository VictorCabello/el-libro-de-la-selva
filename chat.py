from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

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

docs = qdrant.similarity_search(query, k=2)

if docs:
    print(f"Success! Found {len(docs)} relevant chunks.")
    print(f"Top result: {docs[0].page_content}...")
else:
    print("Search returned no results. Ingestion may have failed.")
