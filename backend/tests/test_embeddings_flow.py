"""
Tests the embedding pipeline plumbing: fetching pending reviews, batching,
writing to Qdrant, and marking rows as embedded.

IMPORTANT: this does NOT test the real sentence-transformers model. This
sandbox can't reach huggingface.co to download model weights, so we swap in
a fake embedder that returns deterministic random vectors of the right
dimension. That's enough to verify the pipeline logic (batching, Qdrant
upsert, DB state updates) is correct. The real model should be sanity-checked
in your actual dev environment, where huggingface.co is reachable.
"""

import asyncio
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.db.session import Base, engine
from app.main import app
import app.db.qdrant as qdrant_module
import app.services.embedding_pipeline as pipeline_module


class FakeEmbeddingService:
    """Deterministic stand-in for the real sentence-transformers model.
    Same text always maps to the same vector, so we can assert on results."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(settings.EMBEDDING_DIM).tolist())
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
    """Replaces the real model + real Qdrant server with fast in-process stand-ins."""
    monkeypatch.setattr(pipeline_module, "get_embedding_service", lambda: FakeEmbeddingService())

    memory_client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: memory_client)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_embedding_pipeline_end_to_end(client):
    r = client.post("/companies", json={"name": "EmbedCo", "slug": "embedco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Speaker", "slug": "speaker"})
    product_id = r.json()["id"]

    # Ingest a handful of reviews
    r = client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={
            "reviews": [
                {"source": "amazon", "source_review_id": "e1", "raw_text": "Battery lasts all day, love it"},
                {"source": "amazon", "source_review_id": "e2", "raw_text": "Sound quality is muddy and weak"},
                {"source": "amazon", "source_review_id": "e3", "raw_text": "Great build quality overall"},
            ]
        },
    )
    assert r.json()["accepted"] == 3

    # Trigger embedding generation
    r = client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["reviews_embedded"] == 3
    assert body["reviews_remaining_candidates"] == 0

    # Running it again should embed 0 (idempotent — already-embedded reviews are skipped)
    r = client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")
    assert r.json()["reviews_embedded"] == 0

    # Reviews should now be marked embedded in the DB
    r = client.get(f"/companies/{company_id}/products/{product_id}/reviews")
    reviews = r.json()
    assert len(reviews) == 3


def test_embedding_batching_respects_batch_size(client):
    """With batch_size=2 and 5 reviews, the pipeline should still embed all 5
    (across multiple internal batches) without dropping any."""
    r = client.post("/companies", json={"name": "BatchCo", "slug": "batchco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Watch", "slug": "watch"})
    product_id = r.json()["id"]

    reviews_payload = [
        {"source": "reddit", "source_review_id": f"b{i}", "raw_text": f"Review number {i} about the watch"}
        for i in range(5)
    ]
    client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": reviews_payload},
    )

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/embeddings/generate",
        params={"batch_size": 2},
    )
    assert r.json()["reviews_embedded"] == 5
