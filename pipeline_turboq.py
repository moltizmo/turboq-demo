"""TurboQuant compressed RAG pipeline."""

import time
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from quantizer import TurboQuantizer


class TurboQPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_bits: int = 4):
        self.model = SentenceTransformer(model_name)
        self.quantizer = TurboQuantizer(n_bits=n_bits)
        self.compressed = None
        self._decompressed_cache = None

    def encode_corpus(self, corpus: list[str], batch_size: int = 64, show_progress: bool = True):
        """Encode corpus and compress with TurboQuantizer."""
        print(f"Encoding {len(corpus)} passages for TurboQ...")
        embeddings = self.model.encode(
            corpus, batch_size=batch_size, show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)
        self.compress(embeddings)

    def compress(self, embeddings: np.ndarray):
        """Compress pre-computed embeddings."""
        print(f"Compressing with TurboQuantizer ({self.quantizer.n_bits}-bit)...")
        self.compressed = self.quantizer.quantize(embeddings)
        self._decompressed_cache = None
        print(f"Compressed {embeddings.shape[0]} vectors")

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Search compressed corpus. Returns list of (passage_id, score)."""
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)

        if self._decompressed_cache is None:
            reconstructed = self.quantizer.dequantize(self.compressed)
            reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
            self._decompressed_cache = reconstructed / np.where(norms > 1e-8, norms, 1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = np.nan_to_num(
                (self._decompressed_cache @ q_emb.T).squeeze(),
                nan=-1.0, posinf=-1.0, neginf=-1.0,
            )
        top_k = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]

    def search_timed(self, query: str, k: int = 10) -> tuple[list[tuple[int, float]], float]:
        """Search with timing. Returns (results, latency_ms)."""
        start = time.perf_counter()
        results = self.search(query, k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms

    def memory_mb(self) -> float:
        """Memory footprint of the compressed representation in MB."""
        if self.compressed is None:
            return 0.0
        return self.quantizer.memory_bytes(self.compressed) / (1024 * 1024)
