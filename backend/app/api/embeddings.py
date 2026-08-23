from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Product
from app.services.embedding_pipeline import embed_pending_reviews

router = APIRouter(prefix="/companies/{company_id}/products/{product_id}/embeddings", tags=["embeddings"])


class EmbedResult(BaseModel):
    product_id: str
    reviews_embedded: int
    reviews_remaining_candidates: int


@router.post("/generate", response_model=EmbedResult)
async def generate_embeddings(
    company_id: str,
    product_id: str,
    batch_size: int = 64,
    max_reviews: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Synchronous trigger for embedding generation, meant for demo/sample-data runs.

    At real scale (hundreds of thousands+ reviews) this should be moved behind
    a background task queue instead of blocking an HTTP request — flagged in
    the README, not solved here since we're still on sample data.
    """
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await embed_pending_reviews(
        db, product_id=product_id, batch_size=batch_size, max_reviews=max_reviews
    )
    return EmbedResult(**result)
