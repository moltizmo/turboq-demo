"""TurboQuantizer: Simplified TurboQuant compression for embedding vectors.

Implements PolarQuant (magnitude + quantized direction) and QJL-inspired
sign-bit error correction for near-lossless vector compression.
"""

import warnings
import numpy as np


class TurboQuantizer:
    def __init__(self, n_bits: int = 3, qjl_dim: int = 64, seed: int = 42):
        self.n_bits = n_bits
        self.levels = 2 ** n_bits  # 8 levels for 3-bit
        self.qjl_dim = qjl_dim
        self.rng = np.random.RandomState(seed)
        self._projection = None

    def quantize(self, vectors: np.ndarray) -> dict:
        """PolarQuant: decompose into magnitude + quantized direction."""
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
        directions = vectors / (magnitudes + 1e-8)

        # Quantize direction components from [-1, 1] -> [0, levels-1]
        scale = (self.levels - 1) / 2.0
        quantized = np.clip(
            np.round((directions + 1) * scale), 0, self.levels - 1
        ).astype(np.uint8)

        # QJL sign-bit correction: random projection of residual
        reconstructed_dir = quantized.astype(np.float32) / scale - 1
        residual = directions - reconstructed_dir
        proj = self._get_projection(vectors.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            projected = residual @ proj  # (n, qjl_dim)
        projected = np.nan_to_num(projected, nan=0.0)
        sign_bits = np.packbits((projected > 0).astype(np.uint8), axis=1)

        return {
            "quantized": quantized,
            "magnitudes": magnitudes.astype(np.float16),
            "sign_bits": sign_bits,
        }

    def dequantize(self, data: dict) -> np.ndarray:
        """Reconstruct vectors from quantized representation."""
        scale = (self.levels - 1) / 2.0
        directions = data["quantized"].astype(np.float32) / scale - 1
        magnitudes = data["magnitudes"].astype(np.float32)

        # Apply QJL correction
        sign_bits = np.unpackbits(data["sign_bits"], axis=1)[:, : self.qjl_dim]
        signs = sign_bits.astype(np.float32) * 2 - 1  # map {0,1} -> {-1,+1}
        proj = self._get_projection(directions.shape[1])
        # Approximate residual correction via pseudoinverse-like transpose
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correction = signs @ proj.T * 0.05  # damped correction
        correction = np.nan_to_num(correction, nan=0.0)
        directions = directions + correction

        # Re-normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-8)

        return directions * magnitudes

    def memory_bytes(self, data: dict) -> int:
        """Total memory footprint of compressed representation."""
        return (
            data["quantized"].nbytes
            + data["magnitudes"].nbytes
            + data["sign_bits"].nbytes
        )

    def _get_projection(self, dim: int) -> np.ndarray:
        if self._projection is None or self._projection.shape[0] != dim:
            raw = self.rng.randn(dim, self.qjl_dim).astype(np.float32)
            self._projection = raw / np.sqrt(self.qjl_dim)
        return self._projection
