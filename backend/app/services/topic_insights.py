"""
LLM Insight Generation stage — the last step in the pipeline, and the only
one that calls an LLM at all. Two calls per product, regardless of review
volume:

  1. Topic labeling + summarization: ALL of a product's topics in one call.
     Turns placeholder names like "battery / life / great / charge / long"
     into "Battery Life", plus a summary, complaints, compliments, and
     suggestions per topic — built from a handful of representative review
     snippets per topic, not every review.
  2. Executive summary: one call, built from the topic-level summaries
     produced in step 1 (not raw reviews) — cheap regardless of how many
     reviews or topics exist.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Product, Review, ReviewAspectSentiment, Topic
from app.services.llm_client import LLMGenerationError, generate_json

REPRESENTATIVE_SNIPPETS_PER_SENTIMENT = 3


async def _gather_topic_context(db: AsyncSession, topics: list[Topic]) -> list[dict]:
    """Builds the per-topic context the LLM needs: keywords, counts, and a
    small number of representative positive/negative snippets — not the full
    review set, which is what keeps this call cheap regardless of scale."""
    context = []

    for topic in topics:
        keywords = json.loads(topic.keywords_raw) if topic.keywords_raw else []

        pos_result = await db.execute(
            select(ReviewAspectSentiment)
            .where(ReviewAspectSentiment.topic_id == topic.id, ReviewAspectSentiment.sentiment == "positive")
            .order_by(ReviewAspectSentiment.sentiment_score.desc())
            .limit(REPRESENTATIVE_SNIPPETS_PER_SENTIMENT)
        )
        positive_snippets = [a.snippet for a in pos_result.scalars().all() if a.snippet]

        neg_result = await db.execute(
            select(ReviewAspectSentiment)
            .where(ReviewAspectSentiment.topic_id == topic.id, ReviewAspectSentiment.sentiment == "negative")
            .order_by(ReviewAspectSentiment.sentiment_score.asc())
            .limit(REPRESENTATIVE_SNIPPETS_PER_SENTIMENT)
        )
        negative_snippets = [a.snippet for a in neg_result.scalars().all() if a.snippet]

        context.append(
            {
                "cluster_label": topic.cluster_label,
                "keywords": keywords,
                "review_count": topic.review_count,
                "positive_count": topic.positive_count,
                "negative_count": topic.negative_count,
                "neutral_count": topic.neutral_count,
                "positive_snippets": positive_snippets,
                "negative_snippets": negative_snippets,
            }
        )

    return context


_LABELING_SYSTEM_PROMPT = """You are analyzing customer product review data that has already been \
statistically clustered into topics. Your job is ONLY to interpret and summarize what's already \
been computed — do not invent statistics or counts, use only what's given to you.

Respond with ONLY a JSON array, no markdown fences, no preamble. One object per input topic, \
each shaped exactly like:
{
  "cluster_label": <int, must match the input>,
  "name": "<short human-readable topic name, 1-4 words, e.g. 'Battery Life'>",
  "description": "<one sentence describing what this topic covers>",
  "summary": "<2-3 sentence summary of what customers are saying about this topic>",
  "common_complaints": ["<short phrase>", ...up to 4],
  "common_compliments": ["<short phrase>", ...up to 4],
  "suggestions": ["<short actionable suggestion for the product team>", ...up to 3]
}

If a topic has no negative snippets, common_complaints can be an empty list. Same for compliments \
if there are no positive snippets. Base suggestions only on the complaints actually present."""


async def label_and_summarize_topics(db: AsyncSession, product_id: str) -> dict:
    result = await db.execute(select(Topic).where(Topic.product_id == product_id))
    topics = result.scalars().all()

    if not topics:
        return {
            "product_id": product_id,
            "status": "skipped",
            "reason": "No topics found. Run topic discovery (and ideally sentiment analysis) first.",
            "topics_labeled": 0,
        }

    context = await _gather_topic_context(db, topics)
    user_prompt = json.dumps(context, indent=2)

    try:
        llm_output = generate_json(_LABELING_SYSTEM_PROMPT, user_prompt)
    except LLMGenerationError as e:
        return {"product_id": product_id, "status": "error", "reason": str(e), "topics_labeled": 0}

    if not isinstance(llm_output, list):
        return {
            "product_id": product_id,
            "status": "error",
            "reason": f"Expected a JSON array from the LLM, got {type(llm_output).__name__}",
            "topics_labeled": 0,
        }

    topics_by_cluster_label = {t.cluster_label: t for t in topics}
    topics_labeled = 0

    for entry in llm_output:
        cluster_label = entry.get("cluster_label")
        topic = topics_by_cluster_label.get(cluster_label)
        if topic is None:
            continue  # LLM referenced a cluster_label we didn't send — ignore rather than crash

        topic.name = entry.get("name", topic.name)
        topic.description = entry.get("description")
        topic.ai_summary = entry.get("summary")
        topic.common_complaints = json.dumps(entry.get("common_complaints", []))
        topic.common_compliments = json.dumps(entry.get("common_compliments", []))
        topic.suggestions = json.dumps(entry.get("suggestions", []))

        # Representative reviews: reuse the snippets already gathered rather
        # than asking the LLM to invent/select them.
        matching_context = next((c for c in context if c["cluster_label"] == cluster_label), None)
        if matching_context:
            topic.representative_positive_review = (
                matching_context["positive_snippets"][0] if matching_context["positive_snippets"] else None
            )
            topic.representative_negative_review = (
                matching_context["negative_snippets"][0] if matching_context["negative_snippets"] else None
            )

        topics_labeled += 1

    await db.commit()

    return {"product_id": product_id, "status": "completed", "topics_labeled": topics_labeled}


_EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """You are writing an executive summary of customer feedback for a \
product, based on pre-computed topic-level summaries (not raw reviews). Be concise and business-focused.

Respond with ONLY a JSON object, no markdown fences, no preamble, shaped exactly like:
{
  "executive_summary": "<3-5 sentence executive summary covering overall sentiment, the strongest \
topics, the weakest topics, and any notable emerging concerns, written for a product manager>"
}"""


async def generate_executive_summary(db: AsyncSession, product_id: str) -> dict:
    product = await db.get(Product, product_id)
    if not product:
        return {"product_id": product_id, "status": "error", "reason": "Product not found"}

    result = await db.execute(select(Topic).where(Topic.product_id == product_id))
    topics = result.scalars().all()

    unlabeled = [t for t in topics if not t.ai_summary]
    if not topics:
        return {
            "product_id": product_id,
            "status": "skipped",
            "reason": "No topics found. Run topic discovery first.",
        }
    if unlabeled:
        return {
            "product_id": product_id,
            "status": "skipped",
            "reason": f"{len(unlabeled)} topic(s) have no AI summary yet. Run label_and_summarize_topics first.",
        }

    topic_summaries = [
        {
            "name": t.name,
            "summary": t.ai_summary,
            "review_count": t.review_count,
            "positive_count": t.positive_count,
            "negative_count": t.negative_count,
        }
        for t in topics
    ]

    try:
        llm_output = generate_json(_EXECUTIVE_SUMMARY_SYSTEM_PROMPT, json.dumps(topic_summaries, indent=2))
    except LLMGenerationError as e:
        return {"product_id": product_id, "status": "error", "reason": str(e)}

    if not isinstance(llm_output, dict) or "executive_summary" not in llm_output:
        return {
            "product_id": product_id,
            "status": "error",
            "reason": "LLM response missing expected 'executive_summary' key",
        }

    product.executive_summary = llm_output["executive_summary"]
    product.executive_summary_updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "product_id": product_id,
        "status": "completed",
        "executive_summary": product.executive_summary,
    }
