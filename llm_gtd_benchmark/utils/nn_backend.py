"""
Backend-agnostic Nearest-Neighbor index.

Design rationale
----------------
faiss delivers order-of-magnitude speedups over sklearn for large arrays but
has unreliable Windows pip installation.  Rather than hard-failing or silently
producing wrong results, we expose a single factory function `build_nn_index`
that transparently selects the best available backend at runtime:

    faiss  →  chosen when the library is importable AND the dataset is large
              enough to benefit (≥ 10 000 rows by default), or when the caller
              explicitly requests it.
    sklearn →  always available; used as the safe fallback.

Both backends return (distances, indices) in the same shape and unit (L2
Euclidean distance, not squared), so downstream code is backend-agnostic.

Extending to a new backend (e.g. hnswlib, annoy):
    1. Subclass NNIndex and implement build() + query().
    2. Add a branch in build_nn_index().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# faiss availability probe — done once at import time
# ---------------------------------------------------------------------------
try:
    import faiss as _faiss  # noqa: F401 (used inside _FaissNNIndex)

    _FAISS_AVAILABLE = True
    logger.debug("faiss detected: FaissNNIndex will be used for large datasets.")
except ImportError:
    _FAISS_AVAILABLE = False
    logger.debug(
        "faiss not found; SklearnNNIndex will be used. "
        "Install faiss-cpu for faster KNN on large datasets."
    )

# Minimum number of rows above which faiss is selected in "auto" mode.
_FAISS_AUTO_THRESHOLD: int = 10_000


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class NNIndex(ABC):
    """Abstract nearest-neighbor index.

    All implementations must return Euclidean (L2) distances — not squared.
    """

    @abstractmethod
    def build(self, data: np.ndarray) -> "NNIndex":
        """Fit the index on reference data.

        Parameters
        ----------
        data:   2-D float array of shape (n_reference, n_features).

        Returns self for chaining:  ``idx = NNIndex().build(data)``
        """

    @abstractmethod
    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point.

        Parameters
        ----------
        queries:    2-D float array of shape (n_queries, n_features).
        k:          Number of neighbors to retrieve.

        Returns
        -------
        distances:  (n_queries, k) array of L2 distances (ascending order).
        indices:    (n_queries, k) array of reference-set row indices.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier for logging / reporting."""


# ---------------------------------------------------------------------------
# sklearn backend  (always available)
# ---------------------------------------------------------------------------


class SklearnNNIndex(NNIndex):
    """NNIndex backed by sklearn.neighbors.NearestNeighbors (BallTree / KD-tree).

    Parameters
    ----------
    algorithm:  sklearn NearestNeighbors algorithm ("auto", "ball_tree", "kd_tree", "brute").
    n_jobs:     Number of parallel jobs (-1 = all CPUs).
    """

    def __init__(self, algorithm: str = "auto", n_jobs: int = -1) -> None:
        self._algorithm = algorithm
        self._n_jobs = n_jobs
        self._nn = None  # lazy — instantiated in build()

    @property
    def backend_name(self) -> str:
        return "sklearn"

    def build(self, data: np.ndarray) -> "SklearnNNIndex":
        from sklearn.neighbors import NearestNeighbors

        self._nn = NearestNeighbors(
            algorithm=self._algorithm,
            metric="euclidean",
            n_jobs=self._n_jobs,
        )
        self._nn.fit(data.astype(np.float64))
        return self

    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._nn is None:
            raise RuntimeError("Call build() before query().")
        distances, indices = self._nn.kneighbors(
            queries.astype(np.float64), n_neighbors=k
        )
        return distances.astype(np.float64), indices


# ---------------------------------------------------------------------------
# faiss backend  (optional, best performance)
# ---------------------------------------------------------------------------


class FaissNNIndex(NNIndex):
    """NNIndex backed by faiss.IndexFlatL2 (exact brute-force with BLAS).

    faiss operates exclusively in float32.  Inputs are automatically cast;
    squared L2 distances returned by faiss are converted to L2 distances
    before returning, matching the contract of SklearnNNIndex.
    """

    def __init__(self) -> None:
        self._index = None
        self._d: int = 0

    @property
    def backend_name(self) -> str:
        return "faiss"

    def build(self, data: np.ndarray) -> "FaissNNIndex":
        import faiss

        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        self._d = data_f32.shape[1]
        self._index = faiss.IndexFlatL2(self._d)
        self._index.add(data_f32)
        return self

    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Call build() before query().")
        queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
        sq_distances, indices = self._index.search(queries_f32, k)
        # faiss returns squared L2 → convert to L2 (clamp negatives from float32 rounding)
        distances = np.sqrt(np.maximum(sq_distances, 0.0)).astype(np.float64)
        return distances, indices


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_nn_index(
    data: np.ndarray,
    force_backend: str = "auto",
) -> NNIndex:
    """Construct and fit an NNIndex on *data*, choosing the best backend.

    Parameters
    ----------
    data:
        2-D float array of shape (n_samples, n_features).
    force_backend:
        ``"auto"``    — faiss when available and n_samples ≥ 10 000, else sklearn.
        ``"faiss"``   — always faiss (raises ImportError if not installed).
        ``"sklearn"`` — always sklearn.

    Returns
    -------
    A fitted NNIndex instance.
    """
    if force_backend not in {"auto", "faiss", "sklearn"}:
        raise ValueError(
            f"Unknown backend '{force_backend}'. Choose 'auto', 'faiss', or 'sklearn'."
        )

    if force_backend == "faiss" and not _FAISS_AVAILABLE:
        raise ImportError(
            "force_backend='faiss' requested but faiss is not installed. "
            "Run:  pip install faiss-cpu"
        )

    use_faiss: bool = (
        _FAISS_AVAILABLE
        and force_backend != "sklearn"
        and (force_backend == "faiss" or data.shape[0] >= _FAISS_AUTO_THRESHOLD)
    )

    if use_faiss:
        index = FaissNNIndex().build(data)
    else:
        index = SklearnNNIndex().build(data)

    logger.debug(
        "NNIndex built: backend=%s, n_samples=%d, n_features=%d.",
        index.backend_name,
        data.shape[0],
        data.shape[1],
    )
    return index
