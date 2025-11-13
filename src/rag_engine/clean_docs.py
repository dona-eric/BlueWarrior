import os, re, pathlib
from typing import List, Dict, Union
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

# --- 1. Configuration du chemin et constantes ---

DATA_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "backend/src/data/"

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader, 
    ".txt": TextLoader,
    ".docx": Docx2txtLoader
}


# --- 2. Fonction de Nettoyage d'un seul document ---

def clean_document(file_path: pathlib.Path) -> List[Document]:
    
    """
    Charge et nettoie le contenu d'un seul document.
    
    Retourne une liste d'objets Document de LangChain Core, 
    chacun contenant le texte nettoyé et des métadonnées (source).
    """
    
    file_extension = file_path.suffix.lower()
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        print(f"Fichier ignoré: type non supporté ({file_extension}) pour {file_path.name}")
        return []

    LoaderClass = SUPPORTED_EXTENSIONS[file_extension]
    
    # TextLoader a besoin d'un encodage si on veut être sûr
    if LoaderClass == TextLoader:
        loader = LoaderClass(str(file_path), encoding='utf-8')
    else:
        loader = LoaderClass(str(file_path))

    documents = loader.load()
    
    for doc in documents:
        text = doc.page_content
        
        # 1. Gérer les césures de mots : 'fonction-\nnement' -> 'fonctionnement'
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 2. Remplacer les nouvelles lignes multiples par une seule
        text = re.sub(r'\n+', '\n', text)
        
        # 3. Remplacer les espaces/tabs multiples par un seul espace
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Mettre à jour le contenu de l'objet Document
        doc.page_content = text.strip()
        
        # S'assurer que les métadonnées de source sont claires
        doc.metadata['source'] = file_path.name
        
    return documents

#  Fonction pour parcourir le dossier 

def load_and_clean_all_documents(folder_path: pathlib.Path) -> List[Document]:
    """
    Parcourt tous les fichiers supportés dans le dossier DATA_PATH et applique le nettoyage.
    
    Retourne une liste plate de TOUS les objets Document nettoyés (pages ou fichiers entiers).
    Cette liste est prête pour l'étape de 'chunking'.
    """
    all_documents = []
    
    print(f"Dossier cible pour le chargement: {folder_path.resolve()}")
    
    # rglob cherche récursivement dans les sous-dossiers
    for ext in SUPPORTED_EXTENSIONS.keys():
        for file_path in folder_path.rglob(f"*{ext}"):
            print(f"-> Traitement de : {file_path.name}")
            
            try:
                cleaned_docs = clean_document(file_path)
                all_documents.extend(cleaned_docs)
                
            except Exception as e:
                print(f"Erreur lors du traitement de {file_path.name}: {e}")
                
    return all_documents



if __name__ == "__main__":
    
    if not DATA_PATH.exists():
        print(f" Le dossier de données est introuvable à : {DATA_PATH.resolve()}")
    else:
        print("\n" + "=" * 50)
        # Étape 1: Charger et nettoyer tous les documents
        cleaned_corpus = load_and_clean_all_documents(DATA_PATH)
        
        # Étape 2: Vérification des résultats
        print("\n" + "=" * 50)
        print(f"✅ Nettoyage terminé. {len(cleaned_corpus)} pages/documents prêts pour le chunking.")
        
        if cleaned_corpus:
            # Afficher les informations du premier document traité
            first_doc = cleaned_corpus[0]
            
            print("\n--- Aperçu du Premier Document Nettoyé ---")
            print(f"Source: {first_doc.metadata.get('source', 'Inconnue')}")
            print(f"Extrait (250 premiers caractères):\n{first_doc.page_content[:250]}...")
        else:
            print("Aucun document trouvé ou traité.")