from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Product, Review
from app.schemas.schemas import IngestResult, ReviewBulkCreate, ReviewOut
from app.services.text_cleaning import clean_text, detect_language

router = APIRouter(prefix="/companies/{company_id}/products/{product_id}/reviews", tags=["reviews"])


@router.post("/ingest", response_model=IngestResult, status_code=201)
async def ingest_reviews(
    company_id: str,
    product_id: str,
    payload: ReviewBulkCreate,
    db: AsyncSession = Depends(get_db),
):
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    accepted = 0
    duplicates_skipped = 0

    for review_in in payload.reviews:
        cleaned = clean_text(review_in.raw_text)
        if not cleaned:
            duplicates_skipped += 1  # empty after cleaning, treat as junk/skip
            continue

        language = detect_language(cleaned)

        # source_review_id is what we dedup on when the source provides a stable id.
        # When it's absent (e.g. scraped anonymously), we fall back to allowing the row —
        # true near-duplicate detection (paraphrased spam, copy-pasted reviews) requires
        # embedding similarity and happens later in the pipeline, not at ingest time.
        stmt = (
            pg_insert(Review)
            .values(
                product_id=product_id,
                source=review_in.source,
                source_review_id=review_in.source_review_id,
                raw_text=review_in.raw_text,
                cleaned_text=cleaned,
                language=language,
                rating=review_in.rating,
                author=review_in.author,
                review_date=review_in.review_date,
            )
            .on_conflict_do_nothing(
                index_elements=["product_id", "source", "source_review_id"]
            )
            .returning(Review.id)
        )
        result = await db.execute(stmt)
        if result.first() is not None:
            accepted += 1
        else:
            duplicates_skipped += 1

    await db.commit()

    return IngestResult(
        accepted=accepted,
        duplicates_skipped=duplicates_skipped,
        total_submitted=len(payload.reviews),
    )


@router.get("", response_model=list[ReviewOut])
async def list_reviews(
    company_id: str,
    product_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await db.execute(
        select(Review)
        .where(Review.product_id == product_id, Review.is_duplicate.is_(False))
        .order_by(Review.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()
