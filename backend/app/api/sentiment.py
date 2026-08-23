from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Product
from app.services.aspect_sentiment import analyze_product_sentiment

router = APIRouter(prefix="/companies/{company_id}/products/{product_id}/sentiment", tags=["sentiment"])


@router.post("/analyze")
async def trigger_sentiment_analysis(
    company_id: str,
    product_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Scores sentiment for all pending (not-yet-scored) reviews, per-aspect where
    a review's sentences match multiple discovered topics' keywords, with a
    whole-review fallback otherwise. Requires topics/discover to have run first.
    """
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    return await analyze_product_sentiment(db, product_id=product_id)
