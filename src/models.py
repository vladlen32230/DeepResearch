from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

CHAT_MODEL = ChatOllama(model=os.environ["OLLAMA_CHAT_MODEL_NAME"], top_p=0.95, temperature=0.6)
EMBEDDING_MODEL = OllamaEmbeddings(model=os.environ["OLLAMA_EMBEDDING_MODEL"])

smart_model = OpenAIModel(
    model_name=os.environ["DEEP_RESEARCH_MODEL"],
    provider=OpenAIProvider(
        base_url=os.environ["BASE_URL"], 
        api_key=os.environ["API_KEY"]
    ),
)