# Tests

This directory contains unit tests for the el-libro-de-la-selva project.

## Test Files

- `test_config.py` - Tests for Config dataclass
- `test_loader.py` - Tests for DocumentLoader and DocumentSplitter
- `test_storage.py` - Tests for QdrantStorage
- `test_clustering.py` - Tests for DocumentClusterer, DocumentSummarizer, and DocumentHierarchyBuilder
- `test_prompts.py` - Tests for PromptTemplates
- `test_retrieval.py` - Tests for collapsed_tree_retrieval and tree_traversal_search
- `test_cli.py` - Tests for ingestor CLI
- `test_chat.py` - Tests for chat CLI
- `test_init.py` - Tests for module initialization and exports

## Running Tests

To run all tests:
```bash
./venv/bin/pytest
```

To run tests with coverage:
```bash
./venv/bin/pytest --cov=el_libro_de_la_selva tests/
```

To run a specific test file:
```bash
./venv/bin/pytest tests/test_config.py
```

To run a specific test class:
```bash
./venv/bin/pytest tests/test_loader.py::TestDocumentLoader
```

## Coverage

Current coverage: 99%

The test suite uses mocking to test components in isolation without requiring external services (Ollama, Qdrant).
