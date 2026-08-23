from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# --- Company ---

class CompanyCreate(BaseModel):
    name: str
    slug: str


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    slug: str
    created_at: datetime


# --- Product ---

class ProductCreate(BaseModel):
    name: str
    slug: str
    category: str | None = None


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    company_id: str
    name: str
    slug: str
    category: str | None
    created_at: datetime
    executive_summary: str | None = None
    executive_summary_updated_at: datetime | None = None


# --- Review ---

class ReviewCreate(BaseModel):
    source: str = Field(..., description="e.g. amazon, reddit, play_store")
    source_review_id: str | None = None
    raw_text: str
    rating: float | None = None
    author: str | None = None
    review_date: datetime | None = None


class ReviewBulkCreate(BaseModel):
    reviews: list[ReviewCreate]


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    product_id: str
    source: str
    raw_text: str
    cleaned_text: str | None
    language: str | None
    rating: float | None
    overall_sentiment: str | None
    overall_sentiment_score: float | None
    created_at: datetime


class IngestResult(BaseModel):
    accepted: int
    duplicates_skipped: int
    total_submitted: int


# --- Topic ---

class TopicOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    product_id: str
    name: str
    description: str | None
    review_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    ai_summary: str | None
    common_complaints: str | None = None
    common_compliments: str | None = None
    suggestions: str | None = None
    representative_positive_review: str | None = None
    representative_negative_review: str | None = None


# --- Product analytics ---

class ProductHealthOut(BaseModel):
    product_id: str
    review_count: int
    average_rating: float | None
    overall_sentiment_score: float | None
    topic_count: int


# --- Ask AI ---

class AskAIRequest(BaseModel):
    question: str
    product_id: str | None = None


class AskAIResponse(BaseModel):
    answer: str
    sources: list[str] = []
