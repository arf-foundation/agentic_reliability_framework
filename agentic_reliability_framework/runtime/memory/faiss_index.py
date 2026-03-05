"""
FAISS index wrapper – thread‑safe, with optional text storage.
"""
import logging
import threading
from typing import Optional, Tuple
import numpy as np
import faiss

from .constants import MemoryConstants

logger = logging.getLogger(__name__)


class ProductionFAISSIndex:
    """Thread‑safe FAISS index with optional text storage."""

    def __init__(self, dim: int = MemoryConstants.VECTOR_DIM):
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
        self._lock = threading.RLock()
        self.texts: list = []

    def add(self, vector: np.ndarray) -> int:
        """Add a vector; returns the index ID."""
        with self._lock:
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            self.index.add(vector)
            return int(self.index.ntotal - 1)

    def get_count(self) -> int:
        with self._lock:
            return int(self.index.ntotal)

    def add_text(self, text: str, vector: np.ndarray) -> int:
        """Add text together with its vector."""
        idx = self.add(vector)
        with self._lock:
            # Extend texts list to match index size
            while len(self.texts) <= idx:
                self.texts.append("")
            self.texts[idx] = text
        return idx

    def add_async(self, vector: np.ndarray) -> int:
        """Async‑friendly alias for add()."""
        return self.add(vector)

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) for the query."""
        with self._lock:
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            distances, indices = self.index.search(query_vector, k)
            return distances[0], indices[0]

    def shutdown(self) -> None:
        logger.info("FAISS index shutdown")


def create_faiss_index(dim: int = MemoryConstants.VECTOR_DIM) -> ProductionFAISSIndex:
    return ProductionFAISSIndex(dim)
