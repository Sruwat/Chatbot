from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts):
        return self.model.encode(texts).tolist()
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def get_relevant_chunks(query, vectordb_dir, k=1):
    embedding_function = LocalEmbeddingFunction()
    db = Chroma(persist_directory=vectordb_dir, embedding_function=embedding_function)
    docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

if __name__ == "__main__":
    query = "What is the main topic of the document?"
    results = get_relevant_chunks(query, "vectordb")
    for i, chunk in enumerate(results):
        print(f"Chunk {i+1}:\n{chunk}\n")