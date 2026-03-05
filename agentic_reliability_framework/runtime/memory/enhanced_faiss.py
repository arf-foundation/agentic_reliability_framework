"""
Enhanced FAISS with async search and semantic text search.
"""
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from numpy.typing import NDArray

from .faiss_index import ProductionFAISSIndex
from .constants import MemoryConstants

logger = logging.getLogger(__name__)


class EnhancedFAISSIndex:
    """Adds thread‑safe similarity search and text embedding fallback."""

    def __init__(self, faiss_index: ProductionFAISSIndex):
        self.faiss = faiss_index
        self._lock = faiss_index._lock

    def search(
        self, query_vector: Union[NDArray[np.float32], List[float], np.ndarray], k: int = 5
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Thread‑safe similarity search."""
        with self._lock or self.faiss._lock:
            return self._safe_search(query_vector, k)

    def _safe_search(
        self, query_vector: Union[NDArray[np.float32], List[float], np.ndarray], k: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != MemoryConstants.VECTOR_DIM:
            raise ValueError(f"Dimension mismatch: expected {MemoryConstants.VECTOR_DIM}, got {query_vector.shape[1]}")

        total = self.faiss.get_count()
        if total == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        k = min(k, total)
        distances, indices = self.faiss.index.search(query_vector, k)
        return distances[0], indices[0]

    async def search_async(self, query_vector, k=5):
        """Async version using executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.search, query_vector, k)

    def semantic_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """High‑level semantic search with text query."""
        # For simplicity, fallback to random embedding if no model.
        # In production, integrate sentence-transformers.
        import hashlib
        np.random.seed(int(hashlib.sha256(query_text.encode()).hexdigest()[:8], 16))
        query_embedding = np.random.randn(MemoryConstants.VECTOR_DIM).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.search(query_embedding, k)
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            text = self.faiss.texts[idx] if idx < len(self.faiss.texts) else None
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "similarity": 1.0 / (1.0 + float(dist)),
                "text": text,
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
