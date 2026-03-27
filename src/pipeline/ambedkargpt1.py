# File: src/pipeline/ambedkargpt1.py
import sys
import os
import yaml
import ollama
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"), # Saves logs to file
        logging.StreamHandler()            # Prints to terminal
    ]
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.chunking.semantic_chunker import SemanticChunker
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.retrieval_engine import RetrievalEngine

def initialize_system():
    logging.info("=== Initializing SemRAG System (AmbedkarGPT1) ===")
    
    # Check if data exists, if not, process it
    if not os.path.exists("processed/chunks.pkl"):
        logging.info("[1/3] Running Semantic Chunking (Algorithm 1)...")
        chunker = SemanticChunker()
        chunks = chunker.process()
        with open("processed/chunks.pkl", "wb") as f:
             import pickle
             pickle.dump(chunks, f)

    if not os.path.exists("processed/knowledge_graph.pkl"):
        logging.info("[2/3] Building Knowledge Graph & Communities...")
        kg = KnowledgeGraphBuilder()
        kg.build_graph()
        kg.detect_communities()
        kg.summarize_communities()
        kg.save()
        
    logging.info("[3/3] System Ready. Loading Retrieval Engine...")
    return RetrievalEngine()

def generate_answer(query, engine):
    """
    Combines Local and Global search results and prompts the LLM.
    """
    logging.info(f"Query: {query}")
    
    # 1. Retrieve Context
    logging.info("...Performing Local Search (Equation 4)")
    local_context = engine.local_search(query)
    
    logging.info("...Performing Global Search (Equation 5)")
    global_context = engine.global_search(query)
    
    # 2. Construct Prompt
    context_text = "\n\n--- Local Context (Specific Facts) ---\n" + "\n".join(local_context)
    context_text += "\n\n--- Global Context (Community Summaries) ---\n" + "\n".join(global_context)
    
    prompt = f"""
    You are an expert on Dr. B.R. Ambedkar's works. Use the context below to answer the question.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer (cite the context where possible):
    """
    
    # 3. Generate
    logging.info("...Generating Answer via Ollama")
    response = ollama.chat(model="phi3", messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

if __name__ == "__main__":
    # Ensure processed directory exists
    os.makedirs("processed", exist_ok=True)
    
    engine = initialize_system()
    
    print("\n=== AmbedkarGPT1 Live Demo ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() == 'exit':
            break
            
        answer = generate_answer(user_query, engine)
        print("\n" + "="*50)
        print("RESPONSE:")
        print(answer)
        print("="*50 + "\n")