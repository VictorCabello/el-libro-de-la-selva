# Vector Ingestion Sample - El Libro de la Selva

A self-learning project demonstrating vector database ingestion using text from "El Libro de la Selva" (The Jungle Book). While this example uses a fragment of The Jungle Book for fun, the same approach can be applied to any text source.

## 🚧 Work in Progress

This is an ongoing learning project. The ultimate goal is to develop a chatbot capable of answering questions about the book's content.

## Project Contents

- A fragment of "El Libro de la selva" (not the complete book)
- Vector ingestion script using LangChain and Qdrant
- Basic similarity search example

## Setup

1. Install the package:
   ```bash
   pip install -e .
   ```

2. (Optional) Install dev dependencies for testing:
   ```bash
   pip install -e ".[dev]"
   ```

3. Ensure Ollama and Qdrant are running:
   - Ollama: http://localhost:11434 (with bge-m3 model)
   - Qdrant: http://localhost:6333
     
   💡 **Easy setup**: Check out [n8n-io/self-hosted-ai-starter-kit](https://github.com/n8n-io/self-hosted-ai-starter-kit) for a quick way to get Ollama and Qdrant up and running.

## Usage

### CLI Commands
- **Ingest the text**: `el-libro-de-la-selva-ingestor`
- **Chat**: `el-libro-de-la-selva-chat`

### Running Unit Tests
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=el_libro_de_la_selva
```

### Module Structure
```
el_libro_de_la_selva/
├── cli/            # Command-line interface
│   ├── ingestor.py # Document ingestion CLI
│   └── chat.py     # Chat interface CLI
├── config.py       # Configuration settings
├── loader.py       # Document loading utilities
├── storage.py      # Storage operations
├── retrieval.py    # Retrieval functionality
├── clustering.py   # Hierarchical clustering
└── prompts.py      # Prompt templates
```

## Technologies

- LangChain
- Qdrant (vector database)
- Ollama (embeddings)
