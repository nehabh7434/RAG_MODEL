# File: src/chunking/semantic_chunker.py
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
from pypdf import PdfReader
import os


class SemanticChunker:
    def __init__(self, config_path="config.yaml"):
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.encoder = SentenceTransformer(self.config['models']['embedding_model'])

        try:
            self.nlp = spacy.load(self.config['models']['spacy_model'])
        except OSError:
            print(f"Spacy model '{self.config['models']['spacy_model']}' not found. Downloading...")
            from spacy.cli import download
            download(self.config['models']['spacy_model'])
            self.nlp = spacy.load(self.config['models']['spacy_model'])

        # Hyperparameters
        self.buffer_size = self.config['chunking']['buffer_size']
        self.threshold   = self.config['chunking']['breakpoint_threshold']
        self.max_tokens  = self.config['chunking']['max_tokens']

    # ------------------------------------------------------------------
    def load_pdf(self):
        """Step 1: Load PDF and split into raw sentences page by page."""
        pdf_path = self.config['paths']['pdf_path']
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        reader = PdfReader(pdf_path)
        sentences = []

        # Process page-by-page so spaCy never gets the whole book at once
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                doc = self.nlp(page_text)
                sentences.extend(
                    [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]
                )

        print(f"Extracted {len(sentences)} sentences from PDF.")
        return sentences

    # ------------------------------------------------------------------
    def _buffer_merge(self, sentences):
        """
        Implements 'BufferMerge' from the algorithm.
        Combines a sentence with its neighbours to preserve context for embedding.
        """
        buffered_sentences = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end   = min(len(sentences), i + self.buffer_size + 1)
            combined_text = " ".join(sentences[start:end])
            buffered_sentences.append(combined_text)
        return buffered_sentences

    # ------------------------------------------------------------------
    def _split_chunks_with_overlap(self, chunk_text):
        """
        Handles chunks exceeding token limits.
        Splits >max_tokens into sub-chunks with overlap.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.config['chunking']['overlap_tokens'],
            length_function=len
        )
        return splitter.split_text(chunk_text)

    # ------------------------------------------------------------------
    def process(self):
        """Main execution of the Semantic Chunking Algorithm."""
        print("Loading and splitting PDF...")
        sentences = self.load_pdf()

        if not sentences:
            raise ValueError("No sentences extracted from PDF. Check the file.")

        print("Applying Buffer Merge...")
        buffered_sentences = self._buffer_merge(sentences)

        print("Generating Embeddings...")
        embeddings = self.encoder.encode(buffered_sentences, show_progress_bar=True)

        print("Calculating Cosine Distances and Grouping...")
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(len(embeddings) - 1):
            sim      = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - sim

            if distance < self.threshold:
                # Same topic — keep building current chunk
                current_chunk.append(sentences[i + 1])
            else:
                # Breakpoint found — save chunk and start a new one
                full_chunk_text = " ".join(current_chunk)
                if len(full_chunk_text) > self.max_tokens:
                    sub_chunks = self._split_chunks_with_overlap(full_chunk_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(full_chunk_text)
                current_chunk = [sentences[i + 1]]

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Generated {len(chunks)} semantic chunks.")
        return chunks