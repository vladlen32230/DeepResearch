from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

LIGHT_SEARCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant. Use the following context to answer the question.
The context includes information from a knowledge base (RAG) and recent internet search results.
If you don't know the answer based on the context, say "I don't know"."""
    ),
    HumanMessagePromptTemplate.from_template(
        """RAG context:
{rag_context}

Internet search results:
{internet_results}

Question: {question}"""
    )
])
