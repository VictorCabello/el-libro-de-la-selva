"""Pytest configuration and fixtures for el_libro_de_la_selva tests."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test summary")
    return mock
