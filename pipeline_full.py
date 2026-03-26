"""Full-precision RAG pipeline using FAISS flat index with float32 embeddings."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class FullPrecisionPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = 384
        self.index = None
        self.passages = []
        self.embeddings = None

    def build_index(self, passages: list[str], batch_size: int = 64):
        """Encode passages and build a FAISS inner-product index."""
        self.passages = passages
        print(f"Encoding {len(passages)} passages (float32)...")
        self.embeddings = self.model.encode(
            passages, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.embeddings = self.embeddings.astype(np.float32)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)
        print(f"Index built: {self.index.ntotal} vectors")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k passages for a query."""
        q_emb = self.model.encode(
            [query_text], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "index": int(idx),
                "passage": self.passages[idx],
            })
        return results

    def query_embedding(self, q_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        """Retrieve top-k using a pre-computed query embedding."""
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "index": int(idx),
                "passage": self.passages[idx],
            })
        return results

    def memory_mb(self) -> float:
        """Memory footprint of the index in MB."""
        if self.embeddings is None:
            return 0.0
        return self.embeddings.nbytes / (1024 * 1024)

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.model.encode(
            [query_text], normalize_embeddings=True
        ).astype(np.float32)
