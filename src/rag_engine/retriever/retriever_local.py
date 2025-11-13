# Fichier: retriever_local.py
# ... (vos imports)
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import Pinecone
import os


def build_retriever_local(index_name: str) -> BaseRetriever:
    """Construit un retriever local utilisant Pinecone."""
    
    # 1. Validation des clés manquantes
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY non trouvé dans l'environnement.")
    if not os.getenv("EMBEDDING_MODEL_NAME"):
        raise ValueError("EMBEDDING_MODEL_NAME non trouvé dans l'environnement.")

    try:
        # 2. DÉFINIR L'EMBEDDING EN PREMIER
        embeddings = PineconeEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME"))
        
        # 3. Initialisation du client Pinecone (SDK) pour la gestion de l'index
        pc = Pinecone() 
        
        # 5. Récupération de l'index
        index = pc.Index(index_name)
        
        # 6. Création du VectorStore : L'objet 'embeddings' est maintenant garanti d'exister.
        vectorstore = PineconeVectorStore(
            index=index, 
            embedding=embeddings
        )
        
        retriever_local = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever_local
        
    except Exception as e:
        print(f"Erreur lors de la création du retriever local: {e}")
        raise