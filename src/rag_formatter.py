from pydantic_ai.agent import Agent
from pydantic_ai.settings import ModelSettings
from src.models import smart_model

formatter = Agent(
    smart_model,
    result_type=list[str],
    system_prompt=(
        "You are a helpful assistant, that will take user's question and return a list of "
        "related questions to the user's question for RAG. Answer on this new questions will contain "
        "enough information to answer the main user's question."
    ),
    
    model_settings=ModelSettings(
        temperature=0.6, top_p=0.95
    )
)

def format_rag_questions(question: str) -> list[str]:
    questions = formatter.run_sync(question).data
    return questions