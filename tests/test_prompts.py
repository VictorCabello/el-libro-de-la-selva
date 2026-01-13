"""Unit tests for prompt templates."""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from el_libro_de_la_selva.prompts import PromptTemplates


class TestPromptTemplates:
    """Test cases for PromptTemplates class."""

    def test_summarization_template(self):
        """Test that summarization_template returns correct template string."""
        template = PromptTemplates.summarization_template()
        
        assert isinstance(template, str)
        assert "{context}" in template
        assert "Resume los siguientes fragmentos" in template

    def test_summarization_template_contains_context_placeholder(self):
        """Test that summarization_template contains the context placeholder."""
        template = PromptTemplates.summarization_template()
        
        assert "{context}" in template
        assert template.count("{context}") == 1

    def test_summarization_template_is_in_spanish(self):
        """Test that the summarization template is in Spanish."""
        template = PromptTemplates.summarization_template()
        
        spanish_words = ["Resume", "fragmentos", "párrafo", "cohesivo"]
        for word in spanish_words:
            assert word in template

    def test_create_summarization_chain(self, mock_llm):
        """Test creating a summarization chain with an LLM."""
        chain = PromptTemplates.create_summarization_chain(mock_llm)
        
        assert isinstance(chain, Runnable)

    def test_create_summarization_chain_with_mock_llm(self, mock_llm):
        """Test that the chain can be created and invoked with mock LLM."""
        chain = PromptTemplates.create_summarization_chain(mock_llm)
        
        context = "Test context for summarization."
        result = chain.invoke({"context": context})
        
        assert result is not None
        assert hasattr(result, 'content')

    def test_create_summarization_chain_with_long_context(self, mock_llm):
        """Test chain with long context."""
        chain = PromptTemplates.create_summarization_chain(mock_llm)
        
        context = "This is sentence one. " * 50
        result = chain.invoke({"context": context})
        
        assert result is not None

    def test_create_summarization_chain_with_empty_context(self, mock_llm):
        """Test chain with empty context."""
        chain = PromptTemplates.create_summarization_chain(mock_llm)
        
        result = chain.invoke({"context": ""})
        
        assert result is not None

    def test_create_summarization_chain_with_unicode(self, mock_llm):
        """Test chain with Unicode characters in context."""
        chain = PromptTemplates.create_summarization_chain(mock_llm)
        
        context = "Café 日本語 🦄 Special characters."
        result = chain.invoke({"context": context})
        
        assert result is not None


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test summary")
    return mock
