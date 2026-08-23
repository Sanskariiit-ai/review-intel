"""
Ask AI: answers free-form questions about a product's (or company's) reviews.

Two layers of context, both cheap regardless of review volume:
  1. Topic-level summaries (already computed by the insights stage) — good for
     "what do users dislike most", "compare A and B" style questions.
  2. A handful of semantically-relevant individual review snippets, retrieved
     via Qdrant vector search on the question itself — good for "summarize all
     camera related complaints" style questions that need specific examples.

This is the second (and last) place the LLM gets called from — like the
insights stage, one call per question, never per review.

KNOWN LIMITATIONS (explicitly not solved here):
- No real time-series/trend data yet (no "why is the battery rating
  decreasing" — that needs the trend-analysis dashboard piece, not built).
- "After the latest update" style questions can't be answered precisely
  without a way to define what "the latest update" means (a release-date
  field we don't have) — the LLM will do its best from review dates it can
  see in the retrieved snippets, but this isn't a real filter.
"""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Product, Review, Topic
from app.services.embeddings import get_embedding_service
from app.services.llm_client import LLMGenerationError, generate_text
import app.db.qdrant as qdrant_module

SEMANTIC_SEARCH_TOP_K = 8


async def _gather_topic_context_for_products(db: AsyncSession, products: list[Product]) -> list[dict]:
    context = []
    for product in products:
        result = await db.execute(select(Topic).where(Topic.product_id == product.id))
        topics = result.scalars().all()

        context.append(
            {
                "product_name": product.name,
                "executive_summary": product.executive_summary,
                "topics": [
                    {
                        "name": t.name,
                        "summary": t.ai_summary,
                        "common_complaints": json.loads(t.common_complaints) if t.common_complaints else [],
                        "common_compliments": json.loads(t.common_compliments) if t.common_compliments else [],
                        "positive_count": t.positive_count,
                        "negative_count": t.negative_count,
                        "neutral_count": t.neutral_count,
                    }
                    for t in topics
                ],
            }
        )
    return context


async def _semantic_search_snippets(
    db: AsyncSession, question: str, product_ids: list[str]
) -> list[dict]:
    """Embeds the question and finds the most semantically similar reviews,
    restricted to the given products. Returns empty list gracefully if no
    reviews have been embedded yet — Ask AI should degrade, not crash."""
    if not product_ids:
        return []

    embedding_service = get_embedding_service()
    question_vector = embedding_service.encode([question])[0]

    hits = qdrant_module.search_similar(
        vector=question_vector, product_ids=product_ids, limit=SEMANTIC_SEARCH_TOP_K
    )

    if not hits:
        return []

    review_ids = [hit.payload["review_id"] for hit in hits]
    result = await db.execute(select(Review).where(Review.id.in_(review_ids)))
    reviews_by_id = {r.id: r for r in result.scalars().all()}

    snippets = []
    for hit in hits:
        review = reviews_by_id.get(hit.payload["review_id"])
        if review:
            snippets.append(
                {
                    "text": review.cleaned_text or review.raw_text,
                    "rating": review.rating,
                    "source": review.source,
                    "relevance_score": round(hit.score, 3),
                }
            )
    return snippets


_ASK_AI_SYSTEM_PROMPT = """You are a product intelligence assistant answering questions about \
customer reviews, based ONLY on the context provided (topic summaries and review snippets). \
Do not invent statistics, counts, or claims not supported by the context.

If the question asks about something the context doesn't cover (e.g. a specific time trend that \
isn't represented in the data given), say so plainly rather than guessing.

Write a direct, conversational answer — no markdown headers, no bullet-point dumps unless the \
question specifically calls for a list. A few sentences to a short paragraph is usually right."""


async def answer_question(
    db: AsyncSession,
    question: str,
    company_id: str,
    product_id: str | None = None,
) -> dict:
    if product_id:
        product = await db.get(Product, product_id)
        if not product or product.company_id != company_id:
            return {"status": "error", "reason": "Product not found", "answer": None, "sources": []}
        products = [product]
    else:
        result = await db.execute(select(Product).where(Product.company_id == company_id))
        products = result.scalars().all()
        if not products:
            return {"status": "error", "reason": "No products found for this company", "answer": None, "sources": []}

    topic_context = await _gather_topic_context_for_products(db, products)
    has_any_topics = any(p["topics"] for p in topic_context)

    product_ids = [p.id for p in products]
    snippets = await _semantic_search_snippets(db, question, product_ids)

    if not has_any_topics and not snippets:
        return {
            "status": "skipped",
            "reason": "No processed data available yet (no topics or embedded reviews for these products).",
            "answer": None,
            "sources": [],
        }

    user_prompt = json.dumps(
        {
            "question": question,
            "topic_level_context": topic_context,
            "relevant_review_snippets": snippets,
        },
        indent=2,
    )

    try:
        answer = generate_text(_ASK_AI_SYSTEM_PROMPT, user_prompt)
    except LLMGenerationError as e:
        return {"status": "error", "reason": str(e), "answer": None, "sources": []}

    sources = [p.name for p in products]

    return {"status": "completed", "answer": answer, "sources": sources}
