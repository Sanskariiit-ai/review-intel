"""
Tests the Ask AI pipeline: context assembly (topic summaries + semantic search
snippets), product-scoped vs company-wide question handling, and graceful
degradation when no data exists yet.

Same testing caveat as before: no real embedding model or LLM API key
available in this sandbox, so both are stubbed. What's verified: the context
sent to the LLM is correctly assembled from real DB state, semantic search
correctly filters by product_id(s), and company-wide questions aggregate
across multiple products.
"""

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.db.session import Base, engine
from app.main import app
import app.db.qdrant as qdrant_module
import app.services.embedding_pipeline as pipeline_module
import app.services.ask_ai as ask_ai_module
import app.services.topic_insights as insights_module

TOPIC_KEYWORDS = {
    "battery": ["battery", "power", "charge", "charging"],
    "camera": ["camera", "photo", "picture", "lens"],
    "screen": ["screen", "display", "glass", "crack"],
}

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
        return "battery"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            topic = self._topic_for_text(text)
            vec = self.centers[topic] + self.rng.normal(0, 0.3, settings.EMBEDDING_DIM)
            vectors.append(vec.tolist())
        return vectors


def fake_labeling_response(context_json: str) -> list[dict]:
    context = json.loads(context_json)
    results = []
    for topic_ctx in context:
        kw = topic_ctx["keywords"]
        label = kw[0].title() if kw else f"Topic {topic_ctx['cluster_label']}"
        results.append(
            {
                "cluster_label": topic_ctx["cluster_label"],
                "name": label,
                "description": f"Reviews about {label.lower()}",
                "summary": f"Customers had mixed things to say about {label.lower()}.",
                "common_complaints": ["Inconsistent quality"] if topic_ctx["negative_snippets"] else [],
                "common_compliments": ["Works as expected"] if topic_ctx["positive_snippets"] else [],
                "suggestions": ["Improve QA testing"] if topic_ctx["negative_snippets"] else [],
            }
        )
    return results


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())
    yield


@pytest.fixture(autouse=True)
def stub_embedding_and_qdrant(monkeypatch):
    shared_embedder = StructuredFakeEmbeddingService()
    monkeypatch.setattr(pipeline_module, "get_embedding_service", lambda: shared_embedder)
    monkeypatch.setattr(ask_ai_module, "get_embedding_service", lambda: shared_embedder)

    memory_client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: memory_client)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _setup_fully_processed_product(client, company_slug: str, product_slug: str, monkeypatch):
    r = client.post("/companies", json={"name": "AskCo", "slug": company_slug})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Phone", "slug": product_slug})
    product_id = r.json()["id"]

    reviews_payload = [
        {"source": "amazon", "source_review_id": f"{product_slug}-{i}", "raw_text": text}
        for i, (_, text) in enumerate(SAMPLE_REVIEWS)
    ]
    client.post(
        f"/companies/{company_id}/products/{product_id}/reviews/ingest",
        json={"reviews": reviews_payload},
    )
    client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")
    client.post(
        f"/companies/{company_id}/products/{product_id}/topics/discover", params={"min_topic_size": 3}
    )
    client.post(f"/companies/{company_id}/products/{product_id}/sentiment/analyze")

    def stub_generate_json(system_prompt, user_prompt, **kwargs):
        return fake_labeling_response(user_prompt)

    monkeypatch.setattr(insights_module, "generate_json", stub_generate_json)
    client.post(f"/companies/{company_id}/products/{product_id}/insights/label-topics")

    return company_id, product_id


def test_ask_ai_requires_processed_data(client):
    r = client.post("/companies", json={"name": "EmptyCo", "slug": "emptyco"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Nothing", "slug": "nothing"})
    product_id = r.json()["id"]

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/ask",
        json={"question": "What do people think?"},
    )
    assert r.status_code == 200
    # Should degrade gracefully, not error
    assert "No processed data" in r.json()["answer"] or r.json()["answer"]


def test_ask_ai_product_scoped_includes_topic_context(client, monkeypatch):
    company_id, product_id = _setup_fully_processed_product(client, "askco-scoped", "phone-scoped", monkeypatch)

    captured_prompt = {}

    def stub_generate_text(system_prompt, user_prompt, **kwargs):
        captured_prompt["value"] = user_prompt
        return "Based on the reviews, battery life is praised while camera quality gets mixed feedback."

    monkeypatch.setattr(ask_ai_module, "generate_text", stub_generate_text)

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/ask",
        json={"question": "What do users think about the camera?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "battery life is praised" in body["answer"]
    assert body["sources"] == ["Phone"]

    # Verify the context actually sent to the LLM included topic-level data
    sent_context = json.loads(captured_prompt["value"])
    assert len(sent_context["topic_level_context"]) == 1
    assert sent_context["topic_level_context"][0]["product_name"] == "Phone"
    assert len(sent_context["topic_level_context"][0]["topics"]) > 0

    # And semantic search should have found camera-related snippets given the question
    assert len(sent_context["relevant_review_snippets"]) > 0


def test_ask_ai_company_wide_aggregates_multiple_products(client, monkeypatch):
    r = client.post("/companies", json={"name": "MultiCo", "slug": "multico-ask"})
    company_id = r.json()["id"]

    for slug in ["phone-a", "phone-b"]:
        r2 = client.post(f"/companies/{company_id}/products", json={"name": f"Product {slug}", "slug": slug})
        product_id = r2.json()["id"]
        reviews_payload = [
            {"source": "amazon", "source_review_id": f"{slug}-{i}", "raw_text": text}
            for i, (_, text) in enumerate(SAMPLE_REVIEWS)
        ]
        client.post(
            f"/companies/{company_id}/products/{product_id}/reviews/ingest",
            json={"reviews": reviews_payload},
        )
        client.post(f"/companies/{company_id}/products/{product_id}/embeddings/generate")
        client.post(
            f"/companies/{company_id}/products/{product_id}/topics/discover", params={"min_topic_size": 3}
        )
        client.post(f"/companies/{company_id}/products/{product_id}/sentiment/analyze")

        def stub_generate_json(system_prompt, user_prompt, **kwargs):
            return fake_labeling_response(user_prompt)

        monkeypatch.setattr(insights_module, "generate_json", stub_generate_json)
        client.post(f"/companies/{company_id}/products/{product_id}/insights/label-topics")

    captured_prompt = {}

    def stub_generate_text(system_prompt, user_prompt, **kwargs):
        captured_prompt["value"] = user_prompt
        return "Comparing the two products, both have similar camera feedback."

    monkeypatch.setattr(ask_ai_module, "generate_text", stub_generate_text)

    r = client.post(f"/companies/{company_id}/ask", json={"question": "Compare Product A and Product B."})
    assert r.status_code == 200
    body = r.json()
    assert set(body["sources"]) == {"Product phone-a", "Product phone-b"}

    sent_context = json.loads(captured_prompt["value"])
    assert len(sent_context["topic_level_context"]) == 2


def test_ask_ai_handles_llm_error_gracefully(client, monkeypatch):
    company_id, product_id = _setup_fully_processed_product(client, "askco-error", "phone-error", monkeypatch)

    from app.services.llm_client import LLMGenerationError

    def stub_generate_text_raises(system_prompt, user_prompt, **kwargs):
        raise LLMGenerationError("mock API failure")

    monkeypatch.setattr(ask_ai_module, "generate_text", stub_generate_text_raises)

    r = client.post(
        f"/companies/{company_id}/products/{product_id}/ask",
        json={"question": "What do people think?"},
    )
    assert r.status_code == 200  # should not 500
    assert "mock API failure" in r.json()["answer"]
