from src.models import smart_model
from pydantic_ai import Agent
from src.internet_search import search_internet
from dotenv import load_dotenv
from pydantic_ai import RunContext
from dataclasses import dataclass
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

load_dotenv()

@dataclass
class ResearchResult:
    original_question: str
    sources: list[str]

deepresearch_agent = Agent(
    model=smart_model, 
    result_type=str, 
    deps_type=ResearchResult,
    system_prompt=(
        "You are a helpful assistant that provdes a very detailed information on provided topic. "
        "It contains a short description of the topic. Use internet search tool to gather more information. "
        "You should use internet search tool multiple times."
    ),
    model_settings=ModelSettings(
        temperature=0.6, top_p=0.95
    )
)

@deepresearch_agent.system_prompt
def system_prompt(ctx: RunContext[ResearchResult]) -> str:
    return (
        f"Original user's question: {ctx.deps.original_question}. "
        "Information about next provided topic must answer this user'squestion."
    )

@deepresearch_agent.tool
def browse_internet(ctx: RunContext[ResearchResult], query: str) -> str:
    """
    Use this tool to search the internet for information on the provided topic.
    Use it multiple times. Do not use it more than 8 times.
    If you see new topic, search information for this if it is related to original question.
    """
    print(f'Searching about provided query: {query}')
    source = search_internet(query, num_results=1)
    ctx.deps.sources.append(source)
    return source

def research_topic(topic: str, original_question: str) -> tuple[str, list[str]]:
    """
    Research a specific topic using the deepresearch_agent.
    """
    context = ResearchResult(original_question, [])
    return deepresearch_agent.run_sync(
        topic, 
        deps=context, 
        usage_limits=UsageLimits(request_limit=50)
    ).data, context.sources

