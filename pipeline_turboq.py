"""TurboQuant compressed RAG pipeline using PolarQuant + QJL."""

import warnings
import numpy as np
from quantizer import TurboQuantizer


class TurboQuantPipeline:
    def __init__(self, n_bits: int = 3):
        self.quantizer = TurboQuantizer(n_bits=n_bits)
        self.compressed = None
        self.passages = []
        self._decompressed_cache = None

    def build_index(self, passages: list[str], embeddings: np.ndarray):
        """Compress pre-computed embeddings with TurboQuant."""
        self.passages = passages
        print(f"Compressing {len(passages)} vectors with TurboQuant ({self.quantizer.n_bits}-bit)...")
        self.compressed = self.quantizer.quantize(embeddings)
        self._decompressed_cache = None
        print(f"Compression complete. Memory: {self.memory_mb():.2f} MB")

    def query(self, q_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        """Retrieve top-k using cosine similarity on dequantized vectors."""
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        # Dequantize on-the-fly (or use cache)
        if self._decompressed_cache is None:
            self._decompressed_cache = self.quantizer.dequantize(self.compressed)
        vecs = self._decompressed_cache

        # Normalize for cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        v_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        v_normed = vecs / (v_norms + 1e-8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = (q_norm @ v_normed.T).flatten()
        scores = np.nan_to_num(scores, nan=-1.0)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "score": float(scores[idx]),
                "index": int(idx),
                "passage": self.passages[idx],
            })
        return results

    def memory_mb(self) -> float:
        if self.compressed is None:
            return 0.0
        return self.quantizer.memory_bytes(self.compressed) / (1024 * 1024)

    def clear_cache(self):
        """Clear decompressed vector cache (forces re-dequantize on next query)."""
        self._decompressed_cache = None
