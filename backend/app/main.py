from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, companies, embeddings, insights, products, reviews, sentiment, topics
from app.core.config import settings

app = FastAPI(
    title="AI Product Review Intelligence Platform",
    version="0.1.0",
    description="Base scaffold: ingestion, storage, and topic/sentiment surfacing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(companies.router)
app.include_router(products.router)
app.include_router(reviews.router)
app.include_router(embeddings.router)
app.include_router(topics.router)
app.include_router(sentiment.router)
app.include_router(insights.router)
app.include_router(ask.router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@app.get("/")
async def root():
    return {"message": "Review Intelligence Platform API", "docs": "/docs"}
