"""
Smoke tests for the base scaffold: company/product creation and review
ingestion with dedup. Runs against an in-memory-ish SQLite file so it needs
no external services.

Run with:
    DATABASE_URL="sqlite+aiosqlite:///./_test.db" pytest tests/ -v
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.db.session import Base, engine
from app.main import app


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_company_and_product(client):
    r = client.post("/companies", json={"name": "TestCorp", "slug": "testcorp-flow"})
    assert r.status_code == 201
    company_id = r.json()["id"]

    r = client.post(
        f"/companies/{company_id}/products",
        json={"name": "Gadget", "slug": "gadget", "category": "electronics"},
    )
    assert r.status_code == 201
    assert r.json()["company_id"] == company_id


def test_duplicate_company_slug_rejected(client):
    client.post("/companies", json={"name": "Dup Co", "slug": "dup-slug"})
    r = client.post("/companies", json={"name": "Dup Co Again", "slug": "dup-slug"})
    assert r.status_code == 409


def test_review_ingest_and_dedup(client):
    r = client.post("/companies", json={"name": "ReviewCo", "slug": "reviewco"})
    company_id = r.json()["id"]
    r = client.post(
        f"/companies/{company_id}/products",
        json={"name": "Phone", "slug": "phone"},
    )
    product_id = r.json()["id"]

    ingest_url = f"/companies/{company_id}/products/{product_id}/reviews/ingest"

    # First submission: 2 new reviews
    r = client.post(
        ingest_url,
        json={
            "reviews": [
                {"source": "amazon", "source_review_id": "a1", "raw_text": "Battery is great", "rating": 5},
                {"source": "amazon", "source_review_id": "a2", "raw_text": "Camera is blurry", "rating": 2},
            ]
        },
    )
    assert r.status_code == 201
    assert r.json()["accepted"] == 2
    assert r.json()["duplicates_skipped"] == 0

    # Second submission: one repeat id (a1), one new (a3)
    r = client.post(
        ingest_url,
        json={
            "reviews": [
                {"source": "amazon", "source_review_id": "a1", "raw_text": "Battery is great (resubmitted)", "rating": 5},
                {"source": "amazon", "source_review_id": "a3", "raw_text": "Screen scratches easily", "rating": 3},
            ]
        },
    )
    assert r.status_code == 201
    assert r.json()["accepted"] == 1
    assert r.json()["duplicates_skipped"] == 1

    r = client.get(f"/companies/{company_id}/products/{product_id}/reviews")
    assert len(r.json()) == 3


def test_product_health_reflects_ingested_reviews(client):
    r = client.post("/companies", json={"name": "HealthCo", "slug": "healthco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Widget", "slug": "widget"})
    product_id = r.json()["id"]

    client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": [{"source": "amazon", "source_review_id": "h1", "raw_text": "Great widget", "rating": 4}]},
    )

    r = client.get(f"/companies/{company_id}/products/{product_id}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["review_count"] == 1
    assert body["average_rating"] == 4.0
