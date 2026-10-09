## COMPLETED: Reflection agent, exactly as in LECTURE 11:
##   1) First LLM call  : Qwen (qwen3:0.6b) is the Support Agent -> drafts an answer using the RAG tool.
##   2) Second LLM call : Qwen is the Reviewer -> returns PASS or REVISE with a critique.
##   3) If REVISE       : Qwen is the Reviser -> produces the improved answer.

from langchain_ollama import ChatOllama

from rag import OLLAMA_BASE_URL
from rag_tool import university_it_rag

LLM_MODEL = "qwen3:0.6b"


class ReflectionAgent:
    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

    def support_agent(self, question):
        """First LLM call: Support Agent drafts an answer with retrieval-augmented context."""
        return university_it_rag.invoke(question)

    def reviewer(self, question, draft):
        """Second LLM call: Reviewer decides whether the draft is good enough (PASS/REVISE)."""
        prompt = (
            "You are a strict reviewer of IT support answers. "
            "Check whether the draft answer below answers the user's question using only known "
            "university IT information, and whether it invents any information (for example a phone "
            "number) that is not known to be true. "
            "If the requested information is not available in the knowledge base, saying so is a "
            "CORRECT answer and must be marked PASS; inventing a number or giving irrelevant "
            "instructions is wrong and must be marked REVISE. "
            "Reply with exactly one line starting with PASS or REVISE, followed by a short critique.\n\n"
            f"User question: {question}\n\n"
            f"Draft answer: {draft}\n\n"
            "Review:"
        )
        return self.llm.invoke(prompt).content

    def reviser(self, question, draft, critique):
        """Third LLM call: Reviser improves the draft using the reviewer's critique."""
        prompt = (
            "You are a university IT support assistant improving a draft answer. "
            "Use the critique to fix the answer. Never invent facts or phone numbers. "
            "If the information is not available, state clearly that it is not available "
            "instead of guessing or giving irrelevant instructions.\n\n"
            f"User question: {question}\n\n"
            f"Draft answer: {draft}\n\n"
            f"Critique: {critique}\n\n"
            "Improved answer:"
        )
        return self.llm.invoke(prompt).content

    def run(self, question):
        draft = self.support_agent(question)
        review = self.reviewer(question, draft)

        first_line = review.strip().splitlines()[0] if review.strip() else "PASS"
        verdict = first_line.upper()

        if "REVISE" in verdict:
            final_answer = self.reviser(question, draft, review)
        else:
            final_answer = draft

        return {
            "draft": draft,
            "review": review,
            "verdict": verdict,
            "final": final_answer,
        }


if __name__ == "__main__":
    question = "What exact phone number should I call for the University IT Service Desk?"

    agent = ReflectionAgent()
    result = agent.run(question)

    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print("\n[DRAFT - Support Agent + RAG]")
    print(result["draft"])
    print("\n[REVIEW - Reviewer]")
    print(result["review"])
    print("\n[FINAL ANSWER]")
    print(result["final"])
    print("=" * 60)
