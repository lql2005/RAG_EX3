# SCSE 26 Graded Exercise #3 — RAG

Retrieval-augmented generation (RAG) agent for the university IT policy knowledge base,
built with LangChain + Chroma + Ollama, using **qwen3:0.6b** (not the 8B model).

## Files

| File | Purpose |
|---|---|
| `policy_loader.py` | Provided — loads the single knowledge base file. Do not change. |
| `rag.py` | Core RAG pipeline: embeddings -> Chroma vector store -> similarity search -> qwen3:0.6b generation. |
| `rag_tool.py` | LangChain tool wrapping the RAG pipeline. |
| `agent.py` | Reflection agent (Support Agent -> Reviewer -> Reviser), as in Lecture 11. |
| `check_data.py` | Data integrity check (no Ollama/Chroma required). |

## Setup

```bash
uv pip install -r requirements.txt

# Embedding model (Ollama must be running)
ollama pull nomic-embed-text

# LLM
ollama pull qwen3:0.6b
```

## Run

```bash
python check_data.py   # optional data check
python agent.py        # asks the IT Service Desk phone number question
```

The knowledge base contains no Service Desk phone number, so the agent must answer
that the number is not available instead of inventing one.
