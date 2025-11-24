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
    k: int = Field(default=6, description="Nombre total de documents à retourner")
    local_weight: float = Field(default=1.0)
    web_weight: float = Field(default=0.8)

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
        
        docs_with_score = []

        for rank, doc in enumerate(local_docs, 1):
            doc.metadata["source_type"] = "local"
            docs_with_score.append((doc, self.local_weight / (rank + 60)))  # 60 = k pour éviter division par zéro

        for rank, doc in enumerate(web_docs, 1):
            doc.metadata["source_type"] = "web"
            docs_with_score.append((doc, self.web_weight / (rank + 60)))

        # Tri par score RRF
        docs_with_score.sort(key=lambda x: x[1], reverse=True)

        # Déduplication finale + limite
        seen_content = set()
        final_docs = []
        for doc, _ in docs_with_score:
            content_hash = hash(doc.page_content[:500])  # hash sur début pour éviter faux négatifs
            if content_hash not in seen_content and len(final_docs) < self.k:
                seen_content.add(content_hash)
                final_docs.append(doc)

        logger.info(f"RRF fusion → {len(local_docs)} local + {len(web_docs)} web → {len(final_docs)} finaux")
        return final_docs

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
        k_retriever=4,
        tavily_api_key=TAVILY_API_KEY,
        include_raw_content=True,
        include_domains=["gouv.fr", "sante.fr", "inca.fr", "cancer.fr", "who.int"]
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
    template = """Tu es un assistant médical expert en oncologie, spécialisé dans le cancer de la prostate et les cancers en général.

RÈGLES STRICTES :
- Tu ne réponds QUE sur le cancer (prostate, sein, poumon, etc.) et la prévention associée.
- Si la question est hors sujet (ex: politique, cuisine, blague), réponds : "Je suis spécialisé dans le cancer et la prévention. Je ne peux pas répondre à cette question."
- Toujours citer tes sources à la fin.
- Réponses claires, empathiques, pédagogiques.
- Jamais d'affirmation sans preuve.
- En cas de doute : "Je vous recommande de consulter un médecin."

Question : {question}

Contexte vérifié :
{context}

Réponse structurée :
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
        retriever_web=retriever_web,
        k=6,
        local_weight=1.0,   # priorité au savoir interne validé
        web_weight=0.9
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=model_llm,
        chain_type="stuff",
        retriever=Ensemble_Retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )   

    return rag_chain
