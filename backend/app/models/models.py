import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    products: Mapped[list["Product"]] = relationship(back_populates="company", cascade="all, delete-orphan")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("company_id", "slug", name="uq_product_company_slug"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    executive_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    executive_summary_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    company: Mapped["Company"] = relationship(back_populates="products")
    reviews: Mapped[list["Review"]] = relationship(back_populates="product", cascade="all, delete-orphan")
    topics: Mapped[list["Topic"]] = relationship(back_populates="product", cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("product_id", "source", "source_review_id", name="uq_review_source_dedup"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))

    source: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "amazon", "reddit", "play_store"
    source_review_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)

    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    review_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False)
    sentiment_computed: Mapped[bool] = mapped_column(Boolean, default=False)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    overall_sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)  # positive/negative/neutral
    overall_sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    product: Mapped["Product"] = relationship(back_populates="reviews")
    aspect_sentiments: Mapped[list["ReviewAspectSentiment"]] = relationship(
        back_populates="review", cascade="all, delete-orphan"
    )


class Topic(Base):
    """A discovered semantic topic/cluster for a product (e.g. 'Battery', 'Camera Quality')."""

    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))

    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False)  # raw cluster id from BERTopic/HDBSCAN
    name: Mapped[str] = mapped_column(String(255), nullable=False)  # LLM-assigned human-readable name, e.g. "Battery"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    review_count: Mapped[int] = mapped_column(Integer, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, default=0)

    # JSON-encoded list of top c-TF-IDF keywords from BERTopic, e.g. ["battery","charge","power"].
    # Reused by the aspect-sentiment stage to match sentences to topics via keyword overlap.
    keywords_raw: Mapped[str | None] = mapped_column(Text, nullable=True)

    ai_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON-encoded lists of short strings, e.g. ["Battery drains fast when gaming", ...]
    common_complaints: Mapped[str | None] = mapped_column(Text, nullable=True)
    common_compliments: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggestions: Mapped[str | None] = mapped_column(Text, nullable=True)
    representative_positive_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    representative_negative_review: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    product: Mapped["Product"] = relationship(back_populates="topics")
    keywords: Mapped[list["Keyword"]] = relationship(back_populates="topic", cascade="all, delete-orphan")
    aspect_sentiments: Mapped[list["ReviewAspectSentiment"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class ReviewAspectSentiment(Base):
    """Links a review to a topic (cluster assignment from the topic-discovery step).

    sentiment/sentiment_score are nullable because topic assignment (this row being
    created) and sentiment scoring happen as two separate pipeline stages — see
    Topic Discovery -> Aspect-Based Sentiment in the architecture. A row can exist
    with a topic assignment but no sentiment yet, until the sentiment stage runs.
    """

    __tablename__ = "review_aspect_sentiments"
    __table_args__ = (UniqueConstraint("review_id", "topic_id", name="uq_review_topic_assignment"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"))
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # cluster membership probability from BERTopic

    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)  # positive/negative/neutral
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # -1.0 to 1.0
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)  # the sentence(s) that triggered this aspect

    review: Mapped["Review"] = relationship(back_populates="aspect_sentiments")
    topic: Mapped["Topic"] = relationship(back_populates="aspect_sentiments")


class Keyword(Base):
    """A canonical keyword/phrase for a topic, with its synonym variants merged in."""

    __tablename__ = "keywords"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))

    canonical_term: Mapped[str] = mapped_column(String(255), nullable=False)
    polarity: Mapped[str] = mapped_column(String(16), nullable=False)  # positive/negative
    frequency: Mapped[int] = mapped_column(Integer, default=1)
    variants: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-encoded list of synonym strings

    topic: Mapped["Topic"] = relationship(back_populates="keywords")


class User(Base):
    """Minimal user table for auth. Multi-tenant scoping (which companies a user can access)
    is deliberately left as a TODO — see note in README."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
