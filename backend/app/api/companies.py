from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import Company
from app.schemas.schemas import CompanyCreate, CompanyOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.post("", response_model=CompanyOut, status_code=201)
async def create_company(payload: CompanyCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(Company).where(Company.slug == payload.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Company slug already exists")

    company = Company(name=payload.name, slug=payload.slug)
    db.add(company)
    await db.commit()
    await db.refresh(company)
    return company


@router.get("", response_model=list[CompanyOut])
async def list_companies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).order_by(Company.created_at.desc()))
    return result.scalars().all()


@router.get("/{company_id}", response_model=CompanyOut)
async def get_company(company_id: str, db: AsyncSession = Depends(get_db)):
    company = await db.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company
