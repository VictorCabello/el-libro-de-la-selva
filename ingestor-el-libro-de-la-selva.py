from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

file_path: str = 'El-libro-de-la-selva.txt'
loader = TextLoader(file_path=file_path, encoding='utf-8')

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_documents(docs)

embedings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
)

qdrant = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedings,
    url="http://localhost:6333",
    collection_name="el-libro-de-la-selva",
)
