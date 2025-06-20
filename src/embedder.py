from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts):
        return self.model.encode(texts).tolist()
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

def embed_chunks(chunks_dir, vectordb_dir):
    docs = []
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(chunks_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(page_content=content))
    embedding_function = LocalEmbeddingFunction()
    db = Chroma.from_documents(docs, embedding_function, persist_directory=vectordb_dir)
    db.persist()
    print(f"Saved embeddings to {vectordb_dir}")

if __name__ == "__main__":
    embed_chunks("chunks", "vectordb")