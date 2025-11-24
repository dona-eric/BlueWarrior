from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.retrievers import BaseRetriever
import os, dotenv, logging
from typing import Optional


logger = logging.getLogger("========= RETRIEVER WEB BUILD===========")
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")

def build_search_web(k_retriever: int = 3,
                     tavily_api_key: str=None
                    )-> BaseRetriever:
    """
    Utilise TailvyRetriever pour effectuer une recherche web, 

    Construit un retriever web ultra-performant avec Tavily.

    Tavily = recherche intelligente + résumé automatique + sources fiables.

    Args:
        k_retriever (int): Nombre de résultats à retourner (3 à 6 recommandé)
        tavily_api_key (str, optional): Clé API Tavily (priorité sur .env)
        include_raw_content (bool): Si True, Tavily retourne le contenu complet des pages (très utile pour RAG)
        include_domains/exclude_domains: Filtrer les sources (ex: seulement .gouv.fr)
    
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




    