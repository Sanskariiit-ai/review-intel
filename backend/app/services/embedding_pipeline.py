"""
The actual "reviews -> embeddings -> vector DB" step of the pipeline.

Batches deliberately: at 1M reviews you cannot load everything into memory
and encode it in one call. This processes `batch_size` reviews at a time,
so memory use stays flat regardless of how many total reviews exist.

This is written to be called either from an API endpoint (fine for a demo /
sample-data run) or from a background worker/task queue (what you'd actually
want at real scale, so a slow embedding run doesn't block an HTTP request).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.qdrant import upsert_review_vectors
from app.models.models import Review
from app.services.embeddings import get_embedding_service


async def embed_pending_reviews(
    db: AsyncSession,
    product_id: str,
    batch_size: int = 64,
    max_reviews: int | None = None,
) -> dict:
    """
    Finds reviews for a product that haven't been embedded yet, encodes them,
    stores the vectors in Qdrant, and marks them as embedded in Postgres.

    Returns a summary dict so the caller (API route, worker, or test) knows
    what happened without re-querying.
    """
    embedding_service = get_embedding_service()

    query = (
        select(Review)
        .where(
            Review.product_id == product_id,
            Review.is_duplicate.is_(False),
            Review.is_embedded.is_(False),
        )
        .order_by(Review.created_at.asc())
    )
    if max_reviews:
        query = query.limit(max_reviews)

    result = await db.execute(query)
    pending_reviews = result.scalars().all()

    total_embedded = 0

    for i in range(0, len(pending_reviews), batch_size):
        batch = pending_reviews[i : i + batch_size]

        # Prefer cleaned_text; fall back to raw_text for any row that
        # somehow lacks it (shouldn't happen via the ingest endpoint, but
        # defensive here since this runs unattended at scale).
        texts = [r.cleaned_text or r.raw_text for r in batch]
        vectors = embedding_service.encode(texts)

        points = [
            {
                "id": review.id,
                "vector": vector,
                "payload": {
                    "review_id": review.id,
                    "product_id": review.product_id,
                    "source": review.source,
                },
            }
            for review, vector in zip(batch, vectors)
        ]
        upsert_review_vectors(points)

        for review in batch:
            review.is_embedded = True
            review.qdrant_point_id = review.id

        await db.commit()
        total_embedded += len(batch)

    return {
        "product_id": product_id,
        "reviews_embedded": total_embedded,
        "reviews_remaining_candidates": len(pending_reviews) - total_embedded,
    }
