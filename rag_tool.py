## COMPLETED: LangChain tool that wraps the RAG pipeline so the agent can call it.

from langchain_core.tools import tool

from rag import rag_query


@tool
def university_it_rag(question: str) -> str:
    """Answer a question about university IT policies (Wi-Fi, VPN, email, storage, accounts, printing).
    Retrieves the most relevant policies from the university knowledge base and answers with qwen3:0.6b.
    If the information is not in the knowledge base, the answer says so instead of guessing."""
    result = rag_query(question)

    if result["sources"]:
        source_info = ", ".join(
            f"{s['policy_id']} ({s['topic']})" for s in result["sources"]
        )
        return f"{result['answer']}\n\nSources: {source_info}"

    return result["answer"]
