# Fichier: retriever_local.py
# ... (vos imports)
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import Pinecone
import os, logging

logger = logging.getLogger("|=========RETRIEVER LOCAL BUILD===========|")


def build_retriever_local(index_name: str,
                        pinecone_api_key: str,
                        embedding_model_name: str
                        ) -> BaseRetriever:
    """Construit un retriever local utilisant Pinecone."""
    
    # 1. Validation des clés manquantes
    if not pinecone_api_key or not embedding_model_name:
        logger.error("PINECONE_API_KEY non trouvé dans l'environnement.")
    
    try:
        # 2. DÉFINIR L'EMBEDDING EN PREMIER
        embeddings = PineconeEmbeddings(model=embedding_model_name)
        
        # 3. Initialisation du client Pinecone (SDK) pour la gestion de l'index
        pc = Pinecone(api_key=pinecone_api_key) 
        
        # 4. Récupération de l'index
        index = pc.Index(index_name)
        
        # 5. Création du VectorStore
        vectorstore = PineconeVectorStore(
            index=index, 
            embedding=embeddings
        )
        
        retriever_local = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever_local
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du retriever local: {e}")
        raise