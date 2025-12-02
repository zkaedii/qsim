"""
Vector Embedding System

This module provides advanced vector embedding generation with multiple methods:
- PCA-based embedding
- Autoencoder-style embedding
- Transformer-style attention-based embedding

Also includes similarity computation with multiple metrics.
"""

import time
from collections import defaultdict
from typing import Any, Dict, Union

try:
    import hashlib
except ImportError:
    hashlib = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .security import secure_operation
from .performance import performance_monitor


class VectorEmbeddingGenius:
    """Advanced vector embedding generation with multiple methods."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.cache: Dict[str, "np.ndarray"] = {}
        self.models = {
            "pca": self._pca_embedding,
            "autoencoder": self._autoencoder_embedding,
            "transformer": self._transformer_embedding,
        }
        self.performance_stats: Dict[str, list] = defaultdict(list)

    @secure_operation
    def generate_embedding(
        self, data: Union[str, "np.ndarray"], method: str = "pca"
    ) -> "np.ndarray":
        """Generate vector embedding using specified method."""
        cache_key = self._generate_cache_key(data, method)

        if cache_key in self.cache:
            return self.cache[cache_key]

        if method not in self.models:
            raise ValueError(f"Unknown embedding method: {method}")

        start_time = time.perf_counter()

        if isinstance(data, str):
            processed_data = self._text_to_array(data)
        elif np and isinstance(data, np.ndarray):
            processed_data = data
        else:
            if not np:
                raise ImportError("numpy is required for this operation")
            processed_data = np.array(data)

        embedding = self.models[method](processed_data)

        self.cache[cache_key] = embedding
        if len(self.cache) > 1000:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        execution_time = time.perf_counter() - start_time
        self.performance_stats[method].append(execution_time)

        return embedding

    def _pca_embedding(self, data: "np.ndarray") -> "np.ndarray":
        """Generate PCA-based embedding."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] < 2:
            if not np:
                raise ImportError("numpy is required for this operation")
            data = np.vstack([data, data + np.random.normal(0, 0.01, data.shape)])

        mean = np.mean(data, axis=0)
        centered = data - mean

        cov = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        embedding_dim = min(self.dimension, eigenvectors.shape[1])
        projection_matrix = eigenvectors[:, :embedding_dim]

        embedding = np.dot(centered[0], projection_matrix)

        if len(embedding) < self.dimension:
            embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
        else:
            embedding = embedding[: self.dimension]

        return self._normalize_embedding(embedding)

    def _autoencoder_embedding(self, data: "np.ndarray") -> "np.ndarray":
        """Generate autoencoder-style embedding."""
        if not np:
            raise ImportError("numpy is required for this operation")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        input_dim = data.shape[1] if data.ndim > 1 else len(data)

        np.random.seed(42)
        encoder_weights = np.random.normal(0, 0.1, (input_dim, self.dimension))

        if data.ndim == 1:
            encoding = np.dot(data, encoder_weights)
        else:
            encoding = np.dot(data[0], encoder_weights)

        embedding = np.tanh(encoding)

        return self._normalize_embedding(embedding)

    def _transformer_embedding(self, data: "np.ndarray") -> "np.ndarray":
        """Generate transformer-style attention-based embedding."""
        if not np:
            raise ImportError("numpy is required for this operation")
        if data.ndim == 1:
            sequence = data.reshape(-1, 1)
        else:
            sequence = data

        seq_len = sequence.shape[0]

        position_encoding = np.zeros((seq_len, self.dimension))
        for pos in range(seq_len):
            for i in range(0, self.dimension, 2):
                position_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / self.dimension)))
                if i + 1 < self.dimension:
                    position_encoding[pos, i + 1] = np.cos(
                        pos / (10000 ** (2 * i / self.dimension))
                    )

        np.random.seed(42)
        W_q = np.random.normal(0, 0.1, (sequence.shape[1], self.dimension))
        W_k = np.random.normal(0, 0.1, (sequence.shape[1], self.dimension))
        W_v = np.random.normal(0, 0.1, (sequence.shape[1], self.dimension))

        Q = np.dot(sequence, W_q)
        K = np.dot(sequence, W_k)
        V = np.dot(sequence, W_v)

        attention_scores = np.dot(Q, K.T) / np.sqrt(self.dimension)
        attention_weights = self._softmax(attention_scores)

        attended = np.dot(attention_weights, V)

        attended_with_pos = attended + position_encoding

        embedding = np.mean(attended_with_pos, axis=0)

        return self._normalize_embedding(embedding)

    def _text_to_array(self, text: str) -> "np.ndarray":
        """Convert text to numerical array."""
        char_values = [ord(c) for c in text[:1000]]

        if len(char_values) < 100:
            char_values.extend([0] * (100 - len(char_values)))
        else:
            char_values = char_values[:100]

        if np:
            return np.array(char_values, dtype=np.float32) / 255.0
        raise ImportError("Numpy is required to process text data.")

    def _softmax(self, x: "np.ndarray") -> "np.ndarray":
        """Compute softmax function."""
        if not np:
            return x
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _normalize_embedding(self, embedding: "np.ndarray") -> "np.ndarray":
        """Normalize embedding to unit length."""
        if not np:
            return embedding
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _generate_cache_key(self, data: Any, method: str) -> str:
        """Generate cache key for data and method."""
        data_str = str(data) if isinstance(data, str) else str(data.tolist())
        if hashlib:
            return f"{method}_{hashlib.md5(data_str.encode()).hexdigest()}"
        return f"{method}_{data_str}"

    @performance_monitor
    def compute_similarity(
        self, embedding1: "np.ndarray", embedding2: "np.ndarray", metric: str = "cosine"
    ) -> float:
        """Compute similarity between embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimension")

        if metric == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            return self._euclidean_similarity(embedding1, embedding2)
        elif metric == "manhattan":
            return self._manhattan_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def _cosine_similarity(self, a: "np.ndarray", b: "np.ndarray") -> float:
        """Compute cosine similarity."""
        if not np:
            return 0.0
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _euclidean_similarity(self, a: "np.ndarray", b: "np.ndarray") -> float:
        """Compute euclidean similarity (inverse of distance)."""
        if not np:
            return 0.0
        distance = np.linalg.norm(a - b)
        return float(1.0 / (1.0 + distance))

    def _manhattan_similarity(self, a: "np.ndarray", b: "np.ndarray") -> float:
        """Compute Manhattan similarity."""
        if not np:
            return 0.0
        distance = np.sum(np.abs(a - b))
        return float(1.0 / (1.0 + distance))
