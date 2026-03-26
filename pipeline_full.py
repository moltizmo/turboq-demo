"""Full-precision RAG pipeline using sentence-transformers + FAISS."""

import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class FullPrecisionPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None

    def encode_corpus(self, corpus: list[str], batch_size: int = 64, show_progress: bool = True):
        """Encode all corpus passages and build FAISS index."""
        print(f"Encoding {len(corpus)} passages (float32)...")
        self.embeddings = self.model.encode(
            corpus, batch_size=batch_size, show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        self.embeddings = self.embeddings.astype(np.float32)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Search entire corpus for a query. Returns list of (passage_id, score)."""
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def search_timed(self, query: str, k: int = 10) -> tuple[list[tuple[int, float]], float]:
        """Search with timing. Returns (results, latency_ms)."""
        start = time.perf_counter()
        results = self.search(query, k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms

    def memory_mb(self) -> float:
        """Memory footprint of the stored embeddings in MB."""
        if self.embeddings is None:
            return 0.0
        return self.embeddings.nbytes / (1024 * 1024)
