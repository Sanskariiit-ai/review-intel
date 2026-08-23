from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Product, Topic
from app.schemas.schemas import TopicOut
from app.services.topic_clustering import discover_topics

router = APIRouter(prefix="/companies/{company_id}/products/{product_id}/topics", tags=["topics"])


@router.post("/discover")
async def trigger_topic_discovery(
    company_id: str,
    product_id: str,
    min_topic_size: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs clustering over the product's embedded reviews and (re)persists topics.
    Synchronous for now, same caveat as embeddings/generate — move to a background
    worker once this needs to run against real review volumes instead of samples.
    """
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    return await discover_topics(db, product_id=product_id, min_topic_size=min_topic_size)


@router.get("", response_model=list[TopicOut])
async def list_topics(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await db.execute(
        select(Topic).where(Topic.product_id == product_id).order_by(Topic.review_count.desc())
    )
    return result.scalars().all()
