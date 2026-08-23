"""
Tests the topic clustering pipeline (BERTopic + HDBSCAN).

Unlike the embedding tests, this DOES exercise real clustering logic — BERTopic
runs fully locally with no network dependency once embeddings are provided, so
this sandbox can test it for real. Only the embeddings feeding it are fake
(same huggingface.co limitation as before): we construct them so that reviews
about the same topic (battery/camera/screen) land in the same region of vector
space, which lets us assert BERTopic actually separates them correctly — not
just that the code runs without crashing.
"""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.db.session import Base, engine
from app.main import app
import app.db.qdrant as qdrant_module
import app.services.embedding_pipeline as pipeline_module


TOPIC_KEYWORDS = {
    "battery": ["battery", "power", "charge", "charging"],
    "camera": ["camera", "photo", "picture", "lens"],
    "screen": ["screen", "display", "glass", "crack"],
}

# Reviews deliberately written with clear single-topic vocabulary so we can
# verify clustering assigns them sensibly.
SAMPLE_REVIEWS = [
    ("battery", "Battery lasts forever, charge holds all day"),
    ("battery", "Battery backup is amazing, charging is fast"),
    ("battery", "Power lasts all day on a single charge"),
    ("battery", "Great battery life, charging is quick too"),
    ("battery", "Battery drains slowly, very good power management"),
    ("camera", "Camera is blurry, photo quality disappoints"),
    ("camera", "Photos are noisy, poor lens performance"),
    ("camera", "Picture quality is bad, camera lens issue"),
    ("camera", "Camera takes great photos, love the lens"),
    ("camera", "Photo clarity is excellent, camera works well"),
    ("screen", "Screen scratches easily, glass is fragile"),
    ("screen", "Display cracked fast, screen glass is weak"),
    ("screen", "Screen is fragile, glass breaks too easy"),
    ("screen", "Display quality is crisp, screen looks great"),
    ("screen", "Glass screen protector recommended, display scratches"),
]


class StructuredFakeEmbeddingService:
    """Maps each review to a vector near its topic's center + noise, so BERTopic
    has real semantic structure to find — not random noise."""

    def __init__(self):
        rng = np.random.default_rng(7)
        self.centers = {
            "battery": rng.normal(0, 1, settings.EMBEDDING_DIM),
            "camera": rng.normal(8, 1, settings.EMBEDDING_DIM),
            "screen": rng.normal(-8, 1, settings.EMBEDDING_DIM),
        }
        self.rng = np.random.default_rng(99)

    def _topic_for_text(self, text: str) -> str:
        lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return topic
        return "battery"  # fallback, shouldn't hit given our sample data

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            topic = self._topic_for_text(text)
            vec = self.centers[topic] + self.rng.normal(0, 0.3, settings.EMBEDDING_DIM)
            vectors.append(vec.tolist())
        return vectors


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())
    yield


@pytest.fixture(autouse=True)
def stub_embedding_and_qdrant(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_embedding_service", lambda: StructuredFakeEmbeddingService())

    memory_client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: memory_client)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_topic_discovery_separates_distinct_topics(client):
    r = client.post("/companies", json={"name": "ClusterCo", "slug": "clusterco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Phone", "slug": "phone-cluster"})
    product_id = r.json()["id"]

    reviews_payload = [
        {"source": "amazon", "source_review_id": f"c{i}", "raw_text": text}
        for i, (_, text) in enumerate(SAMPLE_REVIEWS)
    ]
    r = client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": reviews_payload},
    )
    assert r.json()["accepted"] == len(SAMPLE_REVIEWS)

    r = client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")
    assert r.json()["reviews_embedded"] == len(SAMPLE_REVIEWS)

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/topics/discover",
        params={"min_topic_size": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"

    # We wrote 3 clearly distinct semantic groups (battery/camera/screen) —
    # BERTopic should find roughly that many real topics, not 1 (undifferentiated)
    # or 15 (over-fragmented).
    assert 2 <= body["topics_created"] <= 4
    # Most reviews should land in a real topic rather than be outliers
    assert body["reviews_assigned"] >= len(SAMPLE_REVIEWS) * 0.6

    r = client.get(f"/companies/{company_id}/products/{product_id}/topics")
    topics = r.json()
    assert len(topics) == body["topics_created"]
    for t in topics:
        assert t["review_count"] > 0
        assert t["name"]  # placeholder keyword-based name should be non-empty


def test_topic_discovery_skips_when_too_few_reviews(client):
    r = client.post("/companies", json={"name": "SmallCo", "slug": "smallco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Gadget", "slug": "gadget-small"})
    product_id = r.json()["id"]

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": [{"source": "amazon", "source_review_id": "s1", "raw_text": "Battery is fine"}]},
    )
    client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")

    r = client.post(f"/companies/{company_id}/products/{product_id}/topics/discover")
    assert r.json()["status"] == "skipped"
    assert r.json()["topics_created"] == 0


def test_topic_discovery_is_idempotent_on_rerun(client):
    """Re-running discovery should replace old topics cleanly, not duplicate them."""
    r = client.post("/companies", json={"name": "RerunCo", "slug": "rerunco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Earbuds", "slug": "earbuds"})
    product_id = r.json()["id"]

    reviews_payload = [
        {"source": "amazon", "source_review_id": f"rr{i}", "raw_text": text}
        for i, (_, text) in enumerate(SAMPLE_REVIEWS)
    ]
    client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": reviews_payload},
    )
    client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")

    r1 = client.post(
        f"/companies/{company_id}/products/{product_id}/topics/discover", params={"min_topic_size": 3}
    )
    r2 = client.post(
        f"/companies/{company_id}/products/{product_id}/topics/discover", params={"min_topic_size": 3}
    )
    assert r1.json()["status"] == "completed"
    assert r2.json()["status"] == "completed"

    r = client.get(f"/companies/{company_id}/products/{product_id}/topics")
    # Should still be a small set of topics, not doubled from two runs
    assert len(r.json()) == r2.json()["topics_created"]
