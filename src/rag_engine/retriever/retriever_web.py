from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.retrievers import BaseRetriever
import os, dotenv

dotenv.load_dotenv()

TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")

def build_search_web(k_retriever: int = 3)-> BaseRetriever:
    """
    Utilise TailvyRetriever pour effectuer une recherche web, 
    
    """
    retriever_web = TavilySearchAPIRetriever(
        num_results=k_retriever,
        api_key=TAVILY_API_KEY
    )
    return retriever_web