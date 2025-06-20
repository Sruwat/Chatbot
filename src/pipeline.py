from .retriever import get_relevant_chunks
from .generator import generate_answer
from .chunker import get_chunk_info

def answer_query(query, vectordb_dir, stream=False, return_sources=False):
    chunks = get_relevant_chunks(query, vectordb_dir)
    context = "\n".join(chunks)
    answer = generate_answer(context, query, stream=stream)
    if return_sources:
        return answer, chunks
    return answer

def get_source_chunks(chunks_dir, info_only=False):
    info = get_chunk_info(chunks_dir)
    if info_only:
        return info
    # Optionally, load and return actual chunk texts if needed
    return info

if __name__ == "__main__":
    query = "What is the main topic of the document?"
    print(answer_query(query, "vectordb"))