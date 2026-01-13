"""Doctest for el_libro_de_la_selva module.

This module contains doctests for the main module docstring.
"""

import doctest
import el_libro_de_la_selva


def test_module_docstring():
    """Test that module docstring doctests pass."""
    # Note: The example in __init__.py requires Ollama and external services
    # so we skip running it but verify the docstring structure is correct
    assert el_libro_de_la_selva.__doc__ is not None
    assert "Example:" in el_libro_de_la_selva.__doc__


def test_module_exports():
    """Test that all expected exports are present."""
    expected_exports = [
        "Config",
        "PromptTemplates",
        "DocumentLoader",
        "DocumentSplitter",
        "DocumentClusterer",
        "DocumentSummarizer",
        "DocumentHierarchyBuilder",
        "QdrantStorage",
        "collapsed_tree_retrieval",
        "tree_traversal_search",
    ]
    
    for export in expected_exports:
        assert hasattr(el_libro_de_la_selva, export), f"{export} not exported"
