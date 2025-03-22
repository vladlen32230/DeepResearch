from langchain_community.tools import DuckDuckGoSearchResults

from dotenv import load_dotenv

load_dotenv()

def search_internet(query: str, num_results: int) -> str:
    """
    Search the internet using DuckDuckGo for a given query.
    
    Args:
        query (str): The search query
        
    Returns:
        str: Search results
    """
    search_tool = DuckDuckGoSearchResults(num_results=num_results)
    results = search_tool.run(query)
    return results