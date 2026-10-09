## COMPLETED: Core RAG pipeline (Retrieval -> Augmented -> Generation)
## Follows the LECTURE 11 example: Chroma vector store + embedding model + local LLM.
## The only LLM used is qwen3:0.6b (NOT the 8B model from the lecture).

import os

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from policy_loader import load_policy_documents

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:0.6b"
COLLECTION_NAME = "university_support"
TOP_K = 3
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Pin the Ollama server explicitly so an ambient OLLAMA_HOST cannot redirect us.
os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL


def get_embeddings():
    """LangChain interface to the embedding model provided by Ollama."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vector_store():
    """Build the Chroma vector store from the single policy document file."""
    documents = load_policy_documents()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings()
    )

    vector_store.add_documents(documents)
    return vector_store


def retrieve(question, k=TOP_K):
    """Similarity search: return the k documents most similar to the question."""
    vector_store = build_vector_store()
    return vector_store.similarity_search(question, k=k)


def build_context(documents):
    """Join the retrieved document chunks into a single prompt context."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer(question, context):
    """Generation step: qwen3:0.6b answers using ONLY the retrieved context."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

    system = (
        "You are a university IT support assistant. "
        "Answer the user's question using ONLY the university IT policy context provided below. "
        "If the requested information (for example a phone number) is NOT present in the context, "
        "say that the information is not available and do NOT invent it. "
        "Do not provide any irrelevant instructions. Keep the answer short and factual."
    )

    prompt = (
        f"{system}\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )

    return llm.invoke(prompt).content


def rag_query(question, k=TOP_K):
    """End-to-end RAG: retrieve relevant policies, then generate an answer."""
    documents = retrieve(question, k=k)
    context = build_context(documents)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "context": context,
        "sources": [
            {
                "policy_id": doc.metadata.get("policy_id"),
                "topic": doc.metadata.get("topic"),
            }
            for doc in documents
        ],
        "answer": answer,
    }


if __name__ == "__main__":
    question = "What exact phone number should I call for the University IT Service Desk?"
    result = rag_query(question)
    print("QUESTION:", result["question"])
    print("\nRETRIEVED SOURCES:", result["sources"])
    print("\nCONTEXT:\n", result["context"])
    print("\nANSWER:", result["answer"])
