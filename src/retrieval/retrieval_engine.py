# File: src/retrieval/retrieval_engine.py
import pickle
import os
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalEngine:
    def __init__(self, config_path="config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])

        # Thresholds from config (Equation 4)
        self.threshold_e = self.config['retrieval']['local_threshold_e']  # entity similarity
        self.threshold_d = self.config['retrieval']['local_threshold_d']  # chunk similarity
        self.top_k_local = self.config['retrieval']['top_k_local']
        self.top_k_global = self.config['retrieval']['top_k_global']

        # Load saved knowledge graph data
        self._load_graph()

    # ------------------------------------------------------------------
    # Load persisted graph data
    # ------------------------------------------------------------------
    def _load_graph(self):
        graph_path = "processed/knowledge_graph.pkl"
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                "Knowledge graph not found at processed/knowledge_graph.pkl. "
                "Run initialize_system() first."
            )

        with open(graph_path, "rb") as f:
            data = pickle.load(f)

        self.graph               = data.get("graph", None)
        self.chunk_map           = data.get("chunk_map", [])
        self.communities         = data.get("communities", {})
        self.community_summaries = data.get("community_summaries", {})

        print(f"[RetrievalEngine] Loaded {len(self.chunk_map)} chunks, "
              f"{len(self.communities)} communities, "
              f"{len(self.community_summaries)} community summaries.")

    # ------------------------------------------------------------------
    # LOCAL SEARCH  (Equation 4 in the paper)
    # Finds relevant chunks by:
    #   1. Entity similarity  — query vs entities in each chunk
    #   2. Chunk similarity   — query embedding vs chunk embedding
    # ------------------------------------------------------------------
    def local_search(self, query: str) -> list[str]:
        """
        Returns top-k most relevant chunk texts for the query.
        Combines entity-level and chunk-level cosine similarity.
        """
        if not self.chunk_map:
            return ["No chunks available."]

        query_embedding = self.encoder.encode([query])  # shape (1, dim)

        scores = []

        for chunk in self.chunk_map:
            chunk_emb = chunk.get("embedding")
            if chunk_emb is None:
                continue

            # --- Chunk-level similarity ---
            chunk_sim = cosine_similarity(query_embedding, [chunk_emb])[0][0]

            # --- Entity-level similarity ---
            # Encode all entities in the chunk and take max similarity
            entities = chunk.get("entities", [])
            entity_sim = 0.0
            if entities:
                entity_embeddings = self.encoder.encode(entities)
                sims = cosine_similarity(query_embedding, entity_embeddings)[0]
                entity_sim = float(np.max(sims))

            # Combined score (average of both signals)
            combined = (chunk_sim + entity_sim) / 2.0

            # Apply thresholds (Equation 4)
            if entity_sim >= self.threshold_e or chunk_sim >= self.threshold_d:
                scores.append((combined, chunk["text"]))

        # Sort by score descending, return top-k texts
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [text for _, text in scores[:self.top_k_local]]

        if not results:
            return ["No relevant local context found for this query."]

        return results

    # ------------------------------------------------------------------
    # GLOBAL SEARCH  (Equation 5 in the paper)
    # Finds relevant community summaries by comparing query embedding
    # against pre-computed community summary embeddings.
    # ------------------------------------------------------------------
    def global_search(self, query: str) -> list[str]:
        """
        Returns top-k most relevant community summary texts for the query.
        """
        if not self.community_summaries:
            return ["No community summaries available."]

        query_embedding = self.encoder.encode([query])  # shape (1, dim)

        scores = []

        for com_id, data in self.community_summaries.items():
            summary_emb = data.get("embedding")
            summary_text = data.get("summary", "")

            if summary_emb is None or not summary_text:
                continue

            sim = cosine_similarity(query_embedding, [summary_emb])[0][0]
            scores.append((sim, summary_text))

        # Sort by score descending, return top-k summaries
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [text for _, text in scores[:self.top_k_global]]

        if not results:
            return ["No relevant global context found for this query."]

        return results