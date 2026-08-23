"""
Wraps sentence-transformers so the rest of the app just calls `.encode(texts)`
and doesn't care which model or library is underneath. That indirection matters:
swapping to a different embedding model (or an API-based one) later should only
require changing this file.

NOTE ON MODEL LOADING: the model is loaded once, lazily, on first use, then
cached as a module-level singleton. Loading it per-request would be slow
(model load takes a few seconds) and wasteful.
"""

from functools import lru_cache

from app.core.config import settings


class EmbeddingService:
    def __init__(self, model_name: str):
        # Imported lazily so importing this module doesn't force-load torch/
        # sentence-transformers for code paths that never need embeddings
        # (e.g. running the API with only cleaning/ingestion active).
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(settings.EMBEDDING_MODEL)
