from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Company, Product
from app.schemas.schemas import AskAIRequest, AskAIResponse
from app.services.ask_ai import answer_question

router = APIRouter(tags=["ask-ai"])


@router.post("/companies/{company_id}/products/{product_id}/ask", response_model=AskAIResponse)
async def ask_about_product(
    company_id: str,
    product_id: str,
    payload: AskAIRequest,
    db: AsyncSession = Depends(get_db),
):
    """Product-scoped question, e.g. 'What do users dislike most about this product?'"""
    product = await db.get(Product, product_id)
    if not product or product.company_id != company_id:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await answer_question(db, question=payload.question, company_id=company_id, product_id=product_id)
    return AskAIResponse(answer=result["answer"] or result.get("reason", "No answer available."), sources=result["sources"])


@router.post("/companies/{company_id}/ask", response_model=AskAIResponse)
async def ask_about_company(
    company_id: str,
    payload: AskAIRequest,
    db: AsyncSession = Depends(get_db),
):
    """Company-wide question across all products, e.g. 'Compare Product A and Product B.'"""
    company = await db.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    result = await answer_question(db, question=payload.question, company_id=company_id, product_id=None)
    return AskAIResponse(answer=result["answer"] or result.get("reason", "No answer available."), sources=result["sources"])
