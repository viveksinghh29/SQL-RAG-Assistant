"""CPU-based sentence-transformers embedding wrapper with lazy-loaded, process-level model caching for efficient local embeddings."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.core.logging import get_logger

log = get_logger("embedder")


@lru_cache
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    log.info(f"Loading embedding model: {settings.embedding_model_name} on {settings.embedding_device}")
    return SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)


class Embedder:
    """Thin wrapper around the sentence-transformers model.

    Kept as its own class (rather than free functions) so it can be
    swapped for a hosted embedding API later without touching callers —
    they depend on this class's `embed`/`embed_one` methods, not on
    sentence-transformers directly.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns a (len(texts), dim) float32 array,
        L2-normalized so inner product == cosine similarity in FAISS."""
        model = _get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return _get_model().get_sentence_embedding_dimension()
