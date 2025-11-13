# src/rag_engine/pipeline_finak.py
import os
from typing import List
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.rag_engine.retriever.retriever_local import build_retriever_local
from src.rag_engine.retriever.retriever_web import build_search_web
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pinecone import Pinecone

load_dotenv()

"""
    Chargement des ressources d'environnement .env

    Returns:
        _type_: les fichier .env de pinecome et des api de groq
"""

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GROQ_API_KEY = os.getenv("API_GROQ_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")



# ====================================================================
#  CLASSE DE RETRIEVER MANUEL POUR COMBINER LES DEUX RETRIEVERS
# ====================================================================
class CustomEnsembleRetriever(BaseRetriever):
    """
    Ce retriever manuel combine les résultats de deux retrievers.
    Il n'effectue pas de reclassement (RRF) comme le vrai EnsembleRetriever,
    il se contente de fusionner et de dédoublonner les documents.
    """
    retriever_local: BaseRetriever
    retriever_web: BaseRetriever
    
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


def get_rag_chain(index_name: str = INDEX_NAME):
    
    """
    Args: index_name: le nom de l'index Pinecone crée pour la gestion des embeddings
    
    Returns:
    Construis une chaîne RAG complète avec avec Groq , Langchain et pinecone
    et asé sur la recherche du web afin de fournir des réponses précises et contextuelles.
    """
    
    """
        Initialisation du retrievers local et web
    """
    
    retriever_local = build_retriever_local(index_name=index_name)
    retriever_web = build_search_web(k_retriever=3)

    """ 
    Je combine ici les deux retrievers pour bénéficier à la fois des documents locaux
    et des informations actualisées du web.
    """
    
    model_llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0.5,
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
    EnsembleRetriever = CustomEnsembleRetriever(
        retriever_local=retriever_local,
        retriever_web=retriever_web
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=model_llm,
        chain_type="stuff",
        retriever= EnsembleRetriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )   

    return rag_chain
