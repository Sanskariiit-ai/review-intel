from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Product
from app.services.topic_insights import generate_executive_summary, label_and_summarize_topics

router = APIRouter(prefix="/companies/{company_id}/products/{product_id}/insights", tags=["insights"])


@router.post("/label-topics")
async def trigger_topic_labeling(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    """One LLM call covering ALL of the product's topics at once — not one call per topic."""
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    return await label_and_summarize_topics(db, product_id=product_id)


@router.post("/executive-summary")
async def trigger_executive_summary(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    """Requires label-topics to have run first — builds on topic summaries, not raw reviews."""
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    return await generate_executive_summary(db, product_id=product_id)


@router.post("/generate")
async def trigger_full_insights_generation(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    """Convenience endpoint: runs topic labeling then executive summary in sequence."""
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    labeling_result = await label_and_summarize_topics(db, product_id=product_id)
    if labeling_result["status"] != "completed":
        return {"labeling": labeling_result, "executive_summary": None}

    summary_result = await generate_executive_summary(db, product_id=product_id)
    return {"labeling": labeling_result, "executive_summary": summary_result}
