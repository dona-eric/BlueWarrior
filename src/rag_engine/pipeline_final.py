# src/rag_engine/pipeline_final.py
import os, logging, dotenv, pathlib
from typing import List
from pathlib import Path
from pydantic import Field
from dotenv import load_dotenv
from src.rag_engine.retriever.retriever_local import build_retriever_local
from src.rag_engine.retriever.retriever_web import build_search_web
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pinecone import Pinecone
from configparser import ConfigParser
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger("=========== PIPELINE RAG FINALE ===============")


#=================== FONCTION DE LECTURE DE CONFIGURATION ===========================
def get_config() -> ConfigParser:
    """Lit le fichier config.ini."""
    
    config = ConfigParser()
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / 'config.ini'
    try:
        # Lire le fichier dans le répertoire courant
        config.read(config_path) 
        if not config.sections():
             raise FileNotFoundError("Le fichier config.ini n'a pas pu être lu ou est vide.")
        logger.info("Configuration lue avec succès depuis config.ini.")
        return config
    except Exception as e:
        logger.critical(f"Erreur lors de la lecture du fichier de configuration: {e}")
        raise

# ====================================================================
# CLASSE DE RETRIEVER MANUEL POUR COMBINER LES DEUX RETRIEVERS
# ====================================================================
class CustomEnsembleRetriever(BaseRetriever):
    """
    Ce retriever manuel combine les résultats de deux retrievers.
    Il n'effectue pas de reclassement (RRF) comme le vrai EnsembleRetriever,
    il se contente de fusionner et de dédoublonner les documents.
    """
    retriever_local: BaseRetriever = Field(...)
    retriever_web: BaseRetriever = Field(...)

    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str):
        """
        La fonction que Langchain appelle pour obtenir les documents.
        """
        # 1. Récupérer les documents des deux sources
        try:
             local_docs = self.retriever_local.get_relevant_documents(query)
             web_docs = self.retriever_web.get_relevant_documents(query)
        except AttributeError:
             local_docs = self.retriever_local.invoke(query)
             web_docs = self.retriever_web.invoke(query)
        
        # 2. Fusionner les listes
        all_docs = local_docs + web_docs
        
        # 3.# On utilise un dictionnaire pour garder les documents uniques
        # basé sur leur contenu (page_content)
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        
        return list(unique_docs.values())

def get_env_or_config(config, section, key):
    """Retourne la valeur depuis l'environnement si présente, sinon depuis config.ini"""
    return os.environ.get(key) or config.get(section, key, fallback=None)

def get_rag_chain():

    #config = get_config()

    try:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("INDEX_NAME")
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
        GROQ_API_KEY = os.getenv("API_GROQ_KEY")
        LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        HOST = os.getenv("HOST")
        TEMPERATURE = float(os.getenv("TEMPERATURE"))

        logger.info("Les environnements sont chargés avec succès")
    except Exception as e:
        logger.error("Les clés d'environnement ne sont pas chargées.")
        raise e
    """
        Initialisation du retrievers local et web
    """
    
    retriever_local = build_retriever_local(
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        embedding_model_name=EMBEDDING_MODEL_NAME
        )
    
    retriever_web = build_search_web(
        k_retriever=3,
        tavily_api_key=TAVILY_API_KEY
        )

    """ 
    Je combine ici les deux retrievers pour bénéficier à la fois des documents locaux
    et des informations actualisées du web.
    """
    
    model_llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        max_tokens=1024,
        stop=None,
        
    )

    """================ Template de prompt personnalisé ================"""
    template = """
    Tu es un assistant médical spécialisé dans la prévention du cancer de la prostate.
    Réponds avec précision, pédagogie et bienveillance.
    Si tu n’es pas sûr d’une réponse, indique-le clairement.

    Question utilisateur : {question}

    Contexte (documents) :
    {context}

    Réponse :
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=
        ["question",
         "context"
         ]
    )

    """ 
        RAG CHAINE FINALE
    """
    Ensemble_Retriever = CustomEnsembleRetriever(
        retriever_local=retriever_local,
        retriever_web=retriever_web
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=model_llm,
        chain_type="stuff",
        retriever= Ensemble_Retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )   

    return rag_chain
