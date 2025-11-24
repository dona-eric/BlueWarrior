# # requirements supplémentaires à installer :
# # pip install sentence-transformers torch flagembedding

# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import os
# import time
# from src.rag_engine.clean_docs import load_and_clean_all_documents, DATA_PATH
# from src.rag_engine.vectore_store import create_rag_chunks

# load_dotenv()

# # --- Config Pinecone (gratuit jusqu'à 1 million de vecteurs) ---
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # obligatoire mais gratuit
# INDEX_NAME = os.getenv("INDEX_NAME", "rag-local-index")

# # Modèle BGE-M3 : meilleur embedding open-source multilingue (français excellent)
# model_name = "BAAI/bge-m3"   # 1024 dimensions

# # Dimension fixe pour ce modèle
# VECTOR_DIMENSION = 1024

# # Initialisation de l'embedding (exécuté localement sur ton CPU/GPU)
# print("Chargement du modèle d'embedding local BAAI/bge-m3 (1024 dim)...")
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'},  # 'cuda' si tu as une GPU
#     encode_kwargs={'normalize_embeddings': True}
# )

# def create_or_get_index():
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     if INDEX_NAME not in pc.list_indexes().names():
#         print(f"Création de l'index {INDEX_NAME} (1024 dim)...")
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=VECTOR_DIMENSION,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )
#         while not pc.describe_index(INDEX_NAME).status["ready"]:
#             time.sleep(1)
#     else:
#         print(f"Index {INDEX_NAME} déjà existant.")
#     return pc.Index(INDEX_NAME)


# def upsert_with_local_embeddings(chunks):
#     print(f"Vectorisation locale de {len(chunks)} chunks avec {model_name}...")
    
#     # Batching pour ne pas exploser la RAM
#     batch_size = 32
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         print(f"  Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ", end="")
        
#         PineconeVectorStore.from_documents(
#             documents=batch,
#             embedding=embeddings,
#             index_name=INDEX_NAME,
#             namespace="docs-fr"
#         )
#         print("OK")
#         time.sleep(0.2)  # petit pause pour Pinecone gratuit

#     print(f"{len(chunks)} chunks indexés avec succès (embedding local gratuit)")


# if __name__ == "__main__":
#     docs = load_and_clean_all_documents(DATA_PATH)
#     chunks = create_rag_chunks(docs, chunk_size=1000, chunk_overlap=200)
    
#     create_or_get_index()
#     upsert_with_local_embeddings(chunks)
    
#     print("RAG 100 % GRATUIT ET LOCAL PRÊT !")
