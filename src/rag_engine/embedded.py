import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings  # ou un autre backend d'embedding
from src.rag_engine.vectore_store import splitter_docs
from src.rag_engine.clean_docs import load_and_clean_all_documents, DATA_PATH

# --- Chargement des variables d'environnement ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")  # Valeur par défaut
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
VECTOR_DIMENSION = 1024


def create_pinecone_index():
    """Créer un index Pinecone s'il n'existe pas."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Création de l'index Pinecone '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
    else:
        print(f"✅ L'index '{INDEX_NAME}' existe déjà.")
    
    return pc.Index(INDEX_NAME)


def embedding_docs(docs):
    """Vectorise les documents et les envoie vers Pinecone."""
    try:
        index = create_pinecone_index()
        
        # 🔹 Sélection du modèle d'embedding
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        
        # 🔹 Création du vector store
        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embedding,
            index_name=INDEX_NAME
        )
        print("✅ Embedding terminé et sauvegardé dans Pinecone.")
        return vector_store
    
    except Exception as e:
        print(f"❌ Erreur lors de la création du vectore store : {e}")
        return None


if __name__ == "__main__":
    # --- Chargement et découpage des documents ---
    print("📄 Chargement et découpage des documents...")
    docs = load_and_clean_all_documents(DATA_PATH)
    from src.rag_engine.vectore_store import splitter_docs
    chunks = splitter_docs(docs)

    # --- Envoi à Pinecone ---
    print("🚀 Vectorisation et envoi à Pinecone...")
    embedding_docs(chunks)
