"""
Thin wrapper around the Qdrant client so the rest of the app never talks to
Qdrant directly. Keeps collection naming, point-id conventions, and payload
shape in one place.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Lazy singleton. Avoids opening a new connection per request."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    return _client


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Creates the review-embeddings collection if it doesn't exist yet.
    Safe to call repeatedly (idempotent)."""
    client = client or get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )


def upsert_review_vectors(
    points: list[dict],
    client: QdrantClient | None = None,
) -> None:
    """
    points: list of dicts shaped like:
        {"id": review_id, "vector": [...], "payload": {"review_id": ..., "product_id": ..., "company_id": ...}}

    Using the review's own UUID as the Qdrant point id keeps the mapping
    1:1 and trivial to reverse-lookup — no separate id-translation table needed.
    """
    client = client or get_qdrant_client()
    ensure_collection(client)

    qdrant_points = [
        qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points
    ]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=qdrant_points)


def search_similar(
    vector: list[float],
    product_ids: list[str] | None = None,
    limit: int = 10,
    client: QdrantClient | None = None,
):
    """Semantic search used by the Ask-AI feature (and later, clustering diagnostics).

    product_ids: restricts the search to one or more products. Pass a single-item
    list for a product-scoped question, multiple ids for a company-wide question
    across all of that company's products, or None to search everything (rare —
    cross-tenant search essentially never makes sense, callers should almost
    always pass at least the company's own product ids).

    Uses query_points() rather than the older search() method — search() was
    deprecated and then removed in qdrant-client 1.19+ (found via testing, not
    docs: the pinned version here vs. what was actually installed differed,
    which is exactly the kind of drift worth pinning tightly for).
    """
    client = client or get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        # No embeddings have been generated for anything yet — expected early on,
        # not an error. Checked explicitly rather than caught by exception type,
        # since that differs between the in-memory test backend (ValueError) and
        # a real Qdrant server (UnexpectedResponse/404) — a mismatch that bit us
        # once already in this file with the search()->query_points() rename.
        return []

    query_filter = None
    if product_ids:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="product_id", match=qmodels.MatchAny(any=product_ids))]
        )

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return response.points
