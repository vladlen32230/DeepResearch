from langchain_chroma import Chroma
from src.models import EMBEDDING_MODEL
from dotenv import load_dotenv

load_dotenv()

def retrieve_chunks(queries: list[str], n_results: int) -> list[str]:
    """
    Retrieve chunks from Chroma DB using a batch of text queries.
    
    Args:
        queries: List of search query strings
        n_results: Number of chunks to return per query
        
    Returns:
        List of retrieved document chunks
    """
    # Initialize the vector store
    db = Chroma(
        collection_name="text_documents",
        embedding_function=EMBEDDING_MODEL,
        persist_directory="src/chroma_db"
    )
    
    ids = set()
    results = []
    
    for query in queries:
        # Retrieve documents based on similarity search
        docs = db.similarity_search(query, k=n_results)
        for doc in docs:
            # Check if we've already seen this document ID
            if doc.id not in ids:
                # Add this document's ID to our set of seen IDs
                ids.add(doc.id)
                # Add the document content to our results
                results.append(doc.page_content)
    
    return results