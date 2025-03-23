from dotenv import load_dotenv
from src.models import CHAT_MODEL
import os

from src.chroma_queries import retrieve_chunks
from src.internet_search import search_internet
from src.prompts import LIGHT_SEARCH_PROMPT_TEMPLATE
from src.rag_formatter import format_rag_questions
# Load environment variables
load_dotenv()

def answer_question(
        question: str, 
        use_rag: bool,
        use_internet: bool,
        use_advanced_rag: bool
    ) -> str:
    n_chunks = int(os.environ["RAG_CHUNK_RESULTS"])
    n_internet_results = int(os.environ["DUCKDUCKGO_SEARCH_RESULTS"])

    # Initialize empty contexts
    rag_context = ""
    internet_results = ""
    
    # Retrieve relevant chunks from RAG if enabled
    if use_rag:
        if use_advanced_rag:
            questions = format_rag_questions(question) + [question]
        else:
            questions = [question]

        chunks = retrieve_chunks(questions, n_results=n_chunks)
        print(f'Retrieved {len(chunks)} chunks from RAG')
        rag_context = "\n\n".join(chunks)
    
    # Search the internet for additional context if enabled
    if use_internet:
        internet_results = search_internet(question, n_internet_results)
        print(f'Retrieved {n_internet_results} results from DuckDuckGo internet search')
    
    # Format the prompt using the ChatPromptTemplate
    chat_prompt = LIGHT_SEARCH_PROMPT_TEMPLATE.format_messages(
        rag_context=rag_context,
        internet_results=internet_results,
        question=question
    )
    
    # Generate response
    response = CHAT_MODEL.invoke(chat_prompt)

    filename = 'lightresearch.md'
    with open(filename, 'w') as f:
        f.write(
            f'# Answer to: {question}\n\n{response.content}\n\n'
            f'# RAG context:\n\n{rag_context}\n\n'
            f'# Internet results:\n\n{internet_results}\n\n'
        )

        print(f'Chat response saved to {filename}')
    
    return response.content