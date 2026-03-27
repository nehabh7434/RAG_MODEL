import networkx as nx
import spacy
import pickle
import yaml
import os
import community as community_louvain
from sentence_transformers import SentenceTransformer
from groq import Groq


class KnowledgeGraphBuilder:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.nlp = spacy.load(self.config['models']['spacy_model'])
        self.graph = nx.Graph()
        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])
        self.groq_client = self._get_groq_client()

        # Will be populated during build/detect/summarize
        self.chunk_map = []
        self.communities = {}
        self.community_summaries = {}

    # ------------------------------------------------------------------
    # Groq client helper (same pattern as ambedkargpt1.py)
    # ------------------------------------------------------------------
    def _get_groq_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
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

    # ------------------------------------------------------------------
    # Load chunks
    # ------------------------------------------------------------------
    def load_chunks(self):
        if not os.path.exists("processed/chunks.pkl"):
            raise FileNotFoundError("Chunks file not found. Run chunking first.")
        with open("processed/chunks.pkl", "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------
    def extract_entities(self, chunk):
        """Extract Named Entities using SpaCy (Nodes)."""
        doc = self.nlp(chunk)
        entities = [
            ent.text.lower() for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"]
        ]
        return list(set(entities))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(self):
        """Construct graph with Nodes and Edges."""
        chunks = self.load_chunks()
        print("Building Graph Nodes and Edges...")

        self.chunk_map = []

        for i, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk)
            self.chunk_map.append({
                "id": i,
                "text": chunk,
                "entities": entities,
                "embedding": None
            })

            for entity in entities:
                self.graph.add_node(entity)

            for j in range(len(entities)):
                for k in range(j + 1, len(entities)):
                    if self.graph.has_edge(entities[j], entities[k]):
                        self.graph[entities[j]][entities[k]]['weight'] += 1
                    else:
                        self.graph.add_edge(entities[j], entities[k], weight=1)

        # Batch-encode all chunks at once (faster than one-by-one)
        if self.chunk_map:
            texts = [c["text"] for c in self.chunk_map]
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            for i, emb in enumerate(embeddings):
                self.chunk_map[i]["embedding"] = emb

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges.")
        return self.graph

    # ------------------------------------------------------------------
    # Community detection (Louvain — no C++ build tools required)
    # ------------------------------------------------------------------
    def detect_communities(self):
        """Apply Louvain community detection."""
        print("Detecting Communities...")

        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Skipping community detection.")
            self.communities = {}
            return

        partition = community_louvain.best_partition(self.graph)

        self.communities = {}
        for entity, community_id in partition.items():
            self.communities.setdefault(community_id, []).append(entity)

        print(f"Detected {len(self.communities)} communities.")

    # ------------------------------------------------------------------
    # Community summarisation via Groq
    # ------------------------------------------------------------------
    def summarize_communities(self):
        """Generate summaries for each community using Groq."""
        print("Summarizing Communities (this may take a moment)...")
        self.community_summaries = {}

        for com_id, entities in self.communities.items():
            # Skip tiny communities to save API calls & time
            if len(entities) < 3:
                continue

            entity_list = ", ".join(entities[:20])  # cap at 20 to stay within context
            prompt = (
                f"Summarize the relationship between these entities "
                f"in the context of Dr. B.R. Ambedkar's work: {entity_list}"
            )

            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a scholarly assistant specialised in "
                                "Dr. B.R. Ambedkar's writings and historical context. "
                                "Be concise and factual."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=256,   # summaries don't need to be long
                )
                summary = response.choices[0].message.content

                self.community_summaries[com_id] = {
                    "summary": summary,
                    "entities": entities,
                    "embedding": self.encoder.encode(summary)
                }
                print(f"  Community {com_id} summarised ({len(entities)} entities).")

            except Exception as e:
                print(f"  Error summarising community {com_id}: {e}")

        print(f"Summarised {len(self.community_summaries)} communities.")

    # ------------------------------------------------------------------
    # Persist to disk
    # ------------------------------------------------------------------
    def save(self):
        os.makedirs("processed", exist_ok=True)
        data = {
            "graph": self.graph,
            "chunk_map": self.chunk_map,
            "communities": self.communities,
            "community_summaries": self.community_summaries,
        }
        with open("processed/knowledge_graph.pkl", "wb") as f:
            pickle.dump(data, f)
        print("Knowledge Graph saved to processed/knowledge_graph.pkl")