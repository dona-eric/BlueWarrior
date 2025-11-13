from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter_docs(docs):
    """
    Divise les documents déjà nettoyés en chunks pour le RAG.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks créés à partir de {len(docs)} documents.")
    return chunks
