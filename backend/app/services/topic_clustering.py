"""
Topic Discovery stage: turns a product's review embeddings into named clusters
(BERTopic + HDBSCAN under the hood), without ever calling an LLM per-review.

IMPORTANT DESIGN NOTE — topic stability across reruns:
Re-running clustering on an updated review set can produce entirely different
cluster ids/boundaries than the previous run (this is a known, hard problem
with unsupervised clustering — flagged explicitly in earlier discussion).
This base implementation does the simple, honest thing: each run fully
recomputes topics for the product and replaces the old ones. It does NOT
attempt to align new clusters with old ones ("is this still the same Battery
topic as last week?"). That alignment (via topic embedding similarity, or
BERTopic's built-in `.merge_models`) is real follow-up work if trend
analysis over time becomes a requirement — not solved here.

We also don't call an LLM here for topic naming yet — that's the next phase.
For now each topic's `name` is a placeholder built from its top c-TF-IDF
keywords (e.g. "battery / life / great"), which is what BERTopic gives us
directly, no LLM needed. The LLM will rename these to human-friendly labels
("Battery Life") in the labeling phase.
"""

from bertopic import BERTopic
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.qdrant as qdrant_module
from app.core.config import settings
from app.models.models import Review, ReviewAspectSentiment, Topic

MIN_REVIEWS_TO_CLUSTER = 10  # below this, clustering is statistically meaningless


async def discover_topics(
    db: AsyncSession,
    product_id: str,
    min_topic_size: int = 5,
) -> dict:
    """
    Fetches all embedded (non-duplicate) reviews for a product, retrieves their
    vectors from Qdrant, clusters them with BERTopic, and persists the result:
    - one Topic row per discovered cluster (outliers, cluster -1, are skipped)
    - one ReviewAspectSentiment row per review-to-topic assignment (sentiment
      fields left null — that's the next pipeline stage)

    Existing topics/assignments for this product are cleared first (see module
    docstring on why we don't attempt incremental topic alignment yet).
    """
    result = await db.execute(
        select(Review).where(
            Review.product_id == product_id,
            Review.is_duplicate.is_(False),
            Review.is_embedded.is_(True),
        )
    )
    reviews = result.scalars().all()

    if len(reviews) < MIN_REVIEWS_TO_CLUSTER:
        return {
            "product_id": product_id,
            "status": "skipped",
            "reason": f"Only {len(reviews)} embedded reviews; need at least {MIN_REVIEWS_TO_CLUSTER} to cluster meaningfully.",
            "topics_created": 0,
            "reviews_assigned": 0,
        }

    review_ids = [r.id for r in reviews]
    texts = [r.cleaned_text or r.raw_text for r in reviews]

    # Pull the already-computed vectors back out of Qdrant rather than re-embedding —
    # embeddings were the expensive step and we already paid for it in the prior phase.
    qdrant = qdrant_module.get_qdrant_client()
    retrieved = qdrant.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=review_ids,
        with_vectors=True,
    )
    vector_by_id = {str(point.id): point.vector for point in retrieved}

    missing = [rid for rid in review_ids if rid not in vector_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} reviews are marked is_embedded=True but have no vector in Qdrant. "
            "This indicates the embedding step and DB state are out of sync."
        )

    import numpy as np

    embeddings = np.array([vector_by_id[rid] for rid in review_ids])

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        embedding_model=None,  # we always pass precomputed embeddings, never re-embed here
        verbose=False,
    )
    cluster_ids, probabilities = topic_model.fit_transform(texts, embeddings=embeddings)

    # Clear prior state for this product (full recompute — see module docstring)
    await db.execute(
        delete(ReviewAspectSentiment).where(
            ReviewAspectSentiment.topic_id.in_(select(Topic.id).where(Topic.product_id == product_id))
        )
    )
    await db.execute(delete(Topic).where(Topic.product_id == product_id))
    await db.flush()

    topic_info = topic_model.get_topic_info()
    topics_created = 0
    reviews_assigned = 0
    topic_row_by_cluster_id: dict[int, Topic] = {}

    import json

    for _, row in topic_info.iterrows():
        cluster_label = int(row["Topic"])
        if cluster_label == -1:
            continue  # BERTopic's designation for "didn't fit any cluster" — not a real topic

        keywords = [word for word, _ in topic_model.get_topic(cluster_label)][:5]
        placeholder_name = " / ".join(keywords) if keywords else f"Topic {cluster_label}"

        topic_row = Topic(
            product_id=product_id,
            cluster_label=cluster_label,
            name=placeholder_name,
            description=None,
            review_count=int(row["Count"]),
            keywords_raw=json.dumps(keywords),
        )
        db.add(topic_row)
        topic_row_by_cluster_id[cluster_label] = topic_row
        topics_created += 1

    await db.flush()  # get generated Topic ids before creating assignments

    for review, cluster_label, prob in zip(reviews, cluster_ids, probabilities):
        cluster_label = int(cluster_label)
        if cluster_label == -1:
            continue  # unassigned outlier review — not linked to any topic

        topic_row = topic_row_by_cluster_id[cluster_label]
        # HDBSCAN probability can be a scalar or array depending on config; normalize to float
        confidence = float(prob) if not hasattr(prob, "__len__") else float(max(prob))

        db.add(
            ReviewAspectSentiment(
                review_id=review.id,
                topic_id=topic_row.id,
                confidence=confidence,
            )
        )
        reviews_assigned += 1

    await db.commit()

    return {
        "product_id": product_id,
        "status": "completed",
        "topics_created": topics_created,
        "reviews_assigned": reviews_assigned,
        "reviews_unassigned_as_outliers": len(reviews) - reviews_assigned,
    }
