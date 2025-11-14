from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.retrievers import BaseRetriever
import os, dotenv, logging


logger = logging.getLogger("========= RETRIEVER WEB BUILD===========")
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")

def build_search_web(k_retriever: int = 3, tavily_api_key: str=None)-> BaseRetriever:
    """
    Utilise TailvyRetriever pour effectuer une recherche web, 
    
    """
    
    if not tavily_api_key:
        logger.error("Erreur de chargement de l'api de tavily")
        tavily_api_key=TAVILY_API_KEY
        if not tavily_api_key:
            logger.error("la clé api ne se trouve pas l'environnement de configuration")
        raise ValueError("La clé API denTavily doit etre fournie")
    
    retriever_web = TavilySearchAPIRetriever(
            num_results=k_retriever,
            api_key=tavily_api_key
    )
    return retriever_web