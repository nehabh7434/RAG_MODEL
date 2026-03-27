# File: src/pipeline/ambedkargpt1.py
import sys
import os
import yaml
import logging
import pickle

from groq import Groq

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/system.log"),  # Saves logs to file
        logging.StreamHandler()             # Prints to terminal
    ]
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.chunking.semantic_chunker import SemanticChunker
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.retrieval_engine import RetrievalEngine

# ---------------------------------------------------------------------------
# Groq client — reads GROQ_API_KEY from environment (or Streamlit secrets)
# ---------------------------------------------------------------------------
def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Streamlit Cloud: fall back to st.secrets if available
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Set it as an environment variable or in Streamlit secrets."
        )
    return Groq(api_key=api_key)


def initialize_system():
    logging.info("=== Initializing SemRAG System (AmbedkarGPT1) ===")

    os.makedirs("processed", exist_ok=True)

    # --- Semantic Chunking ---
    if not os.path.exists("processed/chunks.pkl"):
        logging.info("[1/3] Running Semantic Chunking (Algorithm 1)...")
        chunker = SemanticChunker()
        chunks = chunker.process()
        with open("processed/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        logging.info(f"      Saved {len(chunks)} chunks to processed/chunks.pkl")
    else:
        logging.info("[1/3] Chunks already exist — skipping chunking.")

    # --- Knowledge Graph ---
    if not os.path.exists("processed/knowledge_graph.pkl"):
        logging.info("[2/3] Building Knowledge Graph & Communities...")
        kg = KnowledgeGraphBuilder()
        kg.build_graph()
        kg.detect_communities()
        kg.summarize_communities()
        kg.save()
        logging.info("      Knowledge graph saved.")
    else:
        logging.info("[2/3] Knowledge graph already exists — skipping build.")

    logging.info("[3/3] System Ready. Loading Retrieval Engine...")
    return RetrievalEngine()


def generate_answer(query: str, engine) -> str:
    """
    Combines Local and Global search results and prompts the LLM via Groq.
    """
    logging.info(f"Query: {query}")

    # 1. Retrieve Context
    logging.info("...Performing Local Search (Equation 4)")
    local_context = engine.local_search(query)

    logging.info("...Performing Global Search (Equation 5)")
    global_context = engine.global_search(query)

    # 2. Construct Prompt
    context_text = (
        "\n\n--- Local Context (Specific Facts) ---\n"
        + "\n".join(local_context)
        + "\n\n--- Global Context (Community Summaries) ---\n"
        + "\n".join(global_context)
    )

    prompt = f"""You are an expert on Dr. B.R. Ambedkar's works. \
Use the context below to answer the question accurately and cite the context where possible.

Context:
{context_text}

Question: {query}

Answer (cite the context where possible):"""

    # 3. Generate via Groq
    logging.info("...Generating Answer via Groq (llama3-8b-8192)")
    client = _get_groq_client()

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",   # same Llama 3 8B — free tier on Groq
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AmbedkarGPT, a scholarly assistant specialised in "
                    "Dr. B.R. Ambedkar's writings, philosophy, and historical context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,          # lower = more factual / less hallucination
        max_tokens=1024,
    )

    answer = chat_completion.choices[0].message.content
    logging.info("...Answer generated successfully.")
    return answer


# ---------------------------------------------------------------------------
# CLI entry-point (local testing only — won't work on Streamlit Cloud)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)

    engine = initialize_system()

    print("\n=== AmbedkarGPT1 Live Demo ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue

        answer = generate_answer(user_query, engine)
        print("\n" + "=" * 50)
        print("RESPONSE:")
        print(answer)
        print("=" * 50 + "\n")