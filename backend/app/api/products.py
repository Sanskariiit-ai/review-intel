from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Company, Product, Review, Topic
from app.schemas.schemas import ProductCreate, ProductHealthOut, ProductOut

router = APIRouter(prefix="/companies/{company_id}/products", tags=["products"])


@router.post("", response_model=ProductOut, status_code=201)
async def create_product(company_id: str, payload: ProductCreate, db: AsyncSession = Depends(get_db)):
    company = await db.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    product = Product(
        company_id=company_id, name=payload.name, slug=payload.slug, category=payload.category
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return product


@router.get("", response_model=list[ProductOut])
async def list_products(company_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product).where(Product.company_id == company_id))
    return result.scalars().all()


@router.get("/{product_id}", response_model=ProductOut)
async def get_product(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.get("/{product_id}/health", response_model=ProductHealthOut)
async def get_product_health(company_id: str, product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    review_count_result = await db.execute(
        select(func.count(Review.id)).where(Review.product_id == product_id, Review.is_duplicate.is_(False))
    )
    review_count = review_count_result.scalar() or 0

    avg_rating_result = await db.execute(
        select(func.avg(Review.rating)).where(Review.product_id == product_id, Review.is_duplicate.is_(False))
    )
    average_rating = avg_rating_result.scalar()

    avg_sentiment_result = await db.execute(
        select(func.avg(Review.overall_sentiment_score)).where(
            Review.product_id == product_id, Review.is_duplicate.is_(False)
        )
    )
    overall_sentiment_score = avg_sentiment_result.scalar()

    topic_count_result = await db.execute(select(func.count(Topic.id)).where(Topic.product_id == product_id))
    topic_count = topic_count_result.scalar() or 0

    return ProductHealthOut(
        product_id=product_id,
        review_count=review_count,
        average_rating=float(average_rating) if average_rating is not None else None,
        overall_sentiment_score=float(overall_sentiment_score) if overall_sentiment_score is not None else None,
        topic_count=topic_count,
    )
