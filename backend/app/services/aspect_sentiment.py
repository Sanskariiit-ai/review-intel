"""
Aspect-Based Sentiment Analysis stage.

Goal (straight from the spec): "The battery is amazing but the camera is
terrible." -> Battery: Positive, Camera: Negative. That needs sentiment scored
per-aspect within a single review, not one sentiment for the whole review.

The topic-discovery stage (BERTopic) assigns each *whole review* to one
cluster — good for "what is this review mostly about" but not multi-aspect on
its own. This stage adds real multi-aspect capability on top of it, using
only traditional NLP (regex sentence splitting + keyword matching + VADER),
per the spec's constraint that the LLM must not be called per review:

  1. Split each review into sentences.
  2. For every topic the product has, check each sentence for keyword overlap
     (using each topic's top BERTopic keywords, persisted at discovery time).
  3. Score matched sentences with VADER; average when multiple sentences
     match the same topic.
  4. Upsert a ReviewAspectSentiment row per (review, topic) match — a single
     review CAN end up linked to multiple topics this way.
  5. Fallback: if a review's sentences don't literally contain any topic's
     keywords (its cluster assignment was based on semantic similarity, not
     shared vocabulary — these can diverge), we still update its original
     whole-review cluster assignment using whole-review sentiment, so no
     review is left permanently unscored.

KNOWN LIMITATION: this is keyword-overlap matching, not true semantic aspect
extraction. "The screen is great but I hate holding this thing" won't catch
an implicit aspect that doesn't share vocabulary with any discovered topic's
keyword list. Closing that gap properly is an LLM or trained-ABSA-model job —
explicitly out of scope for the traditional-ML base per the spec's cost
constraints, and flagged here rather than silently accepted.
"""

import json
import re
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Review, ReviewAspectSentiment, Topic
from app.services.sentence_split import split_sentences
from app.services.sentiment import score_sentiment


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern]:
    return [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords if kw]


async def analyze_product_sentiment(
    db: AsyncSession,
    product_id: str,
    batch_size: int = 200,
) -> dict:
    topics_result = await db.execute(select(Topic).where(Topic.product_id == product_id))
    topics = topics_result.scalars().all()

    if not topics:
        return {
            "product_id": product_id,
            "status": "skipped",
            "reason": "No topics discovered yet for this product. Run topic discovery first.",
            "reviews_processed": 0,
        }

    topic_patterns: dict[str, list[re.Pattern]] = {}
    for topic in topics:
        keywords = json.loads(topic.keywords_raw) if topic.keywords_raw else []
        topic_patterns[topic.id] = _keyword_patterns(keywords)

    pending_result = await db.execute(
        select(Review).where(
            Review.product_id == product_id,
            Review.is_duplicate.is_(False),
            Review.sentiment_computed.is_(False),
        )
    )
    pending_reviews = pending_result.scalars().all()

    if not pending_reviews:
        return {
            "product_id": product_id,
            "status": "completed",
            "reason": "No pending reviews to score.",
            "reviews_processed": 0,
        }

    reviews_processed = 0
    aspect_assignments_written = 0
    reviews_matched_multiple_aspects = 0

    for batch_start in range(0, len(pending_reviews), batch_size):
        batch = pending_reviews[batch_start : batch_start + batch_size]
        batch_ids = [r.id for r in batch]

        existing_result = await db.execute(
            select(ReviewAspectSentiment).where(ReviewAspectSentiment.review_id.in_(batch_ids))
        )
        existing_by_key: dict[tuple[str, str], ReviewAspectSentiment] = {
            (a.review_id, a.topic_id): a for a in existing_result.scalars().all()
        }
        existing_review_topic_ids: dict[str, list[str]] = defaultdict(list)
        for (review_id, topic_id) in existing_by_key:
            existing_review_topic_ids[review_id].append(topic_id)

        for review in batch:
            text = review.cleaned_text or review.raw_text
            overall_label, overall_score = score_sentiment(text)
            review.overall_sentiment = overall_label
            review.overall_sentiment_score = overall_score
            review.sentiment_computed = True
            reviews_processed += 1

            sentences = split_sentences(text)
            # matches[topic_id] = list of compound scores from sentences matching that topic
            matches: dict[str, list[tuple[str, float]]] = defaultdict(list)

            for sentence in sentences:
                for topic_id, patterns in topic_patterns.items():
                    if any(p.search(sentence) for p in patterns):
                        _, compound = score_sentiment(sentence)
                        matches[topic_id].append((sentence, compound))

            if len(matches) > 1:
                reviews_matched_multiple_aspects += 1

            if matches:
                for topic_id, sentence_scores in matches.items():
                    avg_compound = sum(s for _, s in sentence_scores) / len(sentence_scores)
                    if avg_compound >= 0.05:
                        label = "positive"
                    elif avg_compound <= -0.05:
                        label = "negative"
                    else:
                        label = "neutral"

                    key = (review.id, topic_id)
                    if key in existing_by_key:
                        row = existing_by_key[key]
                        row.sentiment = label
                        row.sentiment_score = avg_compound
                        row.snippet = sentence_scores[0][0]
                    else:
                        db.add(
                            ReviewAspectSentiment(
                                review_id=review.id,
                                topic_id=topic_id,
                                sentiment=label,
                                sentiment_score=avg_compound,
                                snippet=sentence_scores[0][0],
                            )
                        )
                    aspect_assignments_written += 1
            else:
                # Fallback: no keyword match anywhere. If this review already has a
                # cluster assignment from the discovery stage, score it using the
                # whole-review sentiment so it isn't left permanently unscored.
                for topic_id in existing_review_topic_ids.get(review.id, []):
                    row = existing_by_key[(review.id, topic_id)]
                    row.sentiment = overall_label
                    row.sentiment_score = overall_score
                    aspect_assignments_written += 1

        await db.commit()

    await _recompute_topic_aggregates(db, [t.id for t in topics])

    return {
        "product_id": product_id,
        "status": "completed",
        "reviews_processed": reviews_processed,
        "aspect_assignments_written": aspect_assignments_written,
        "reviews_matched_multiple_aspects": reviews_matched_multiple_aspects,
    }


async def _recompute_topic_aggregates(db: AsyncSession, topic_ids: list[str]) -> None:
    """Recomputes each topic's review_count/positive_count/negative_count/neutral_count
    from the ReviewAspectSentiment rows now that sentiment has been scored."""
    for topic_id in topic_ids:
        result = await db.execute(
            select(ReviewAspectSentiment).where(ReviewAspectSentiment.topic_id == topic_id)
        )
        assignments = result.scalars().all()

        topic = await db.get(Topic, topic_id)
        if not topic:
            continue

        topic.review_count = len(assignments)
        topic.positive_count = sum(1 for a in assignments if a.sentiment == "positive")
        topic.negative_count = sum(1 for a in assignments if a.sentiment == "negative")
        topic.neutral_count = sum(1 for a in assignments if a.sentiment == "neutral")

    await db.commit()
