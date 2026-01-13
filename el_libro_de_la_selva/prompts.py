from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class PromptTemplates:
    """Provides reusable prompt templates for LLM operations.

    This class contains static methods for creating prompt templates and
    chains used throughout the document processing pipeline.
    """

    @staticmethod
    def summarization_template() -> str:
        """Return the summarization prompt template.

        The template is in Spanish and instructs the LLM to create a cohesive
        single-paragraph summary from multiple text fragments.

        Returns:
            String template with {context} placeholder for document content
        """
        return "Resume los siguientes fragmentos de texto en un solo párrafo cohesivo:\n\n{context}"

    @staticmethod
    def create_summarization_chain(llm) -> Runnable:
        """Create a summarization chain for use with an LLM.

        This method constructs a LangChain chain that:
        1. Formats the prompt template with the context
        2. Invokes the LLM to generate a summary

        Args:
            llm: LangChain LLM object (e.g., ChatOllama)

        Returns:
            Runnable chain that can be invoked with {"context": document_text}

        Example:
            >>> llm = ChatOllama(model="llama3.2", temperature=0)
            >>> chain = PromptTemplates.create_summarization_chain(llm)
            >>> result = chain.invoke({"context": "Text to summarize..."})
            >>> print(result.content)
            Generated summary...

        Note:
            The returned chain is a LangChain LCEL (LangChain Expression Language)
            chain that can be composed with other chains.
        """
        template = PromptTemplates.summarization_template()
        prompt = ChatPromptTemplate.from_template(template=template)
        return prompt | llm
