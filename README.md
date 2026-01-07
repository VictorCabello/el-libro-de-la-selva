# Vector Ingestion Sample - El Libro de la Selva

A self-learning project demonstrating vector database ingestion using text from "El Libro de la Selva" (The Jungle Book). While this example uses a fragment of The Jungle Book for fun, the same approach can be applied to any text source.

## 🚧 Work in Progress

This is an ongoing learning project. The ultimate goal is to develop a chatbot capable of answering questions about the book's content.

## Project Contents

- A fragment of "El Libro de la selva" (not the complete book)
- Vector ingestion script using LangChain and Qdrant
- Basic similarity search example

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Ollama and Qdrant are running:
   - Ollama: http://localhost:11434 (with bge-m3 model)
   - Qdrant: http://localhost:6333
   
   💡 **Easy setup**: Check out [n8n-io/self-hosted-ai-starter-kit](https://github.com/n8n-io/self-hosted-ai-starter-kit) for a quick way to get Ollama and Qdrant up and running.

## Usage

- **Ingest the text**: `python ingestor-el-libro-de-la-selva.py`
- **Search**: `python chat.py`

## Technologies

- LangChain
- Qdrant (vector database)
- Ollama (embeddings)
