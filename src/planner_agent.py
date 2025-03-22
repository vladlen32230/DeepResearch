from src.models import smart_model
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from dotenv import load_dotenv

load_dotenv()

planner_agent = Agent(
    model=smart_model, 
    result_type=list[str], 
    system_prompt=(
        "You are a helpful assistant that creates a list of titles with short description, "
        "that will be used in final reports consisting of this topics "
        "containing a detailed answer to user's question."
    ),
    
    model_settings=ModelSettings(
        temperature=0.6, top_p=0.95
    )
)

@planner_agent.tool_plain
def receive_human_feedback(report: str):
    """
    Always use this tool to receive human feedback on the titles and their descriptions.
    If human gives positive feedback, you must return structured response on the next step.
    If human gives negative feedback, you must refine the plan and receive feedback again.
    """

    print(report)
    feedback = input("Enter your feedback: ")
    return feedback

def plan_topics(user_query: str) -> list[str]:
    """
    Plan a list of topics on the user's question.
    """
    return planner_agent.run_sync(user_query).data

