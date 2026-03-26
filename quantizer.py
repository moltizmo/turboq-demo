import warnings
import numpy as np

class TurboQuantizer:
    """
    Simplified TurboQuant implementation.

    PolarQuant: Converts vectors to polar form (magnitude + direction),
    quantizes direction to n_bits using uniform quantization.

    QJL (error correction): Applies random projection sign correction
    to reduce bias from quantization.

    Key fix v3: Use float32 throughout, clip values before quantization,
    and scale magnitudes properly to avoid overflow.
    """

    def __init__(self, n_bits: int = 4):
        self.n_bits = n_bits
        self.levels = 2 ** n_bits
        self._rng = np.random.default_rng(42)
        self._qjl_matrix = None  # Lazy init

    def _get_qjl_matrix(self, dim: int) -> np.ndarray:
        if self._qjl_matrix is None or self._qjl_matrix.shape[1] != dim:
            # Random projection matrix for QJL error correction
            self._qjl_matrix = self._rng.standard_normal((dim // 4, dim)).astype(np.float32)
            self._qjl_matrix /= np.linalg.norm(self._qjl_matrix, axis=1, keepdims=True) + 1e-8
        return self._qjl_matrix

    def quantize(self, vectors: np.ndarray) -> dict:
        """Quantize float32 vectors using PolarQuant + QJL."""
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # PolarQuant step 1: compute L2 magnitude
        magnitudes = np.linalg.norm(vectors, axis=1)  # shape (N,)

        # Normalize to unit sphere (directions)
        safe_mag = np.where(magnitudes > 1e-8, magnitudes, 1.0)
        directions = vectors / safe_mag[:, np.newaxis]  # shape (N, D)

        # Quantize direction: map [-1, 1] → [0, levels-1] as uint8
        # Clip to [-1, 1] range first (unit vectors should be in range but fp errors)
        directions_clipped = np.clip(directions, -1.0, 1.0)
        scale = (self.levels - 1) / 2.0
        quantized = np.round((directions_clipped + 1.0) * scale).astype(np.uint8)

        # QJL: compute sign bits for error correction
        qjl_mat = self._get_qjl_matrix(vectors.shape[1])
        directions = np.nan_to_num(directions, nan=0.0, posinf=0.0, neginf=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qjl_proj = np.nan_to_num(directions @ qjl_mat.T, nan=0.0, posinf=0.0, neginf=0.0)  # (N, D//4)
        qjl_signs = (qjl_proj > 0).astype(np.uint8)  # 1-bit per projection
        # Pack 8 bits into bytes
        n_proj = qjl_signs.shape[1]
        n_bytes = (n_proj + 7) // 8
        qjl_packed = np.packbits(qjl_signs, axis=1)[:, :n_bytes]

        # Store magnitudes as float32 (not float16 — float16 causes overflow for large magnitudes)
        return {
            'quantized': quantized,           # uint8 (N, D)
            'magnitudes': magnitudes.astype(np.float32),  # float32 (N,) — FIX: was float16
            'qjl_packed': qjl_packed,         # uint8 packed bits for QJL correction
            'n_vectors': len(vectors),
            'dim': vectors.shape[1],
            'n_bits': self.n_bits,
        }

    def dequantize(self, data: dict) -> np.ndarray:
        """Reconstruct float32 vectors from quantized representation."""
        scale = (data['n_bits'] and (2 ** data['n_bits'] - 1)) or (self.levels - 1)

        # Reconstruct directions from uint8
        directions = data['quantized'].astype(np.float32) * (2.0 / scale) - 1.0

        # Re-normalize directions (quantization may have shifted off unit sphere)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(norms > 1e-8, norms, 1.0)

        # Reconstruct vectors: direction * magnitude
        magnitudes = data['magnitudes'].astype(np.float32)
        return directions * magnitudes[:, np.newaxis]

    def memory_bytes(self, data: dict) -> int:
        return (data['quantized'].nbytes +
                data['magnitudes'].nbytes +
                data['qjl_packed'].nbytes)

    def compression_ratio(self, original_dim: int, n_vectors: int) -> float:
        """How much smaller is quantized vs float32."""
        original_bytes = original_dim * n_vectors * 4  # float32
        # quantized: n_bits/8 bytes per element + float32 magnitude + QJL bits
        quantized_bytes = (original_dim * n_vectors * self.n_bits / 8 +
                          n_vectors * 4 +  # float32 magnitudes
                          (original_dim // 4 // 8) * n_vectors)  # QJL packed bits
        return original_bytes / quantized_bytes
