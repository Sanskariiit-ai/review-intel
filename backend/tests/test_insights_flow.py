"""
Tests the LLM insight-generation pipeline: prompt construction, JSON parsing,
and DB writes. Uses a stub in place of the real Anthropic call, since this
sandbox has no ANTHROPIC_API_KEY configured (same class of limitation as the
huggingface.co restriction for embeddings — the real call needs to be
sanity-checked once you add your key).

What IS verified here: the plumbing is correct end-to-end — context gathering
(keywords + representative snippets), the shape of data sent to the LLM,
correctly mapping the LLM's JSON response back onto the right Topic rows by
cluster_label, error handling for malformed responses, and the two-call
sequencing (executive summary requires labeling to have run first).
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
import app.services.topic_insights as insights_module
from app.services.llm_client import LLMGenerationError

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
    """Stands in for the real LLM: given the topic context we sent, returns a
    plausible-shaped labeling response so we can verify the write-back logic."""
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
    monkeypatch.setattr(pipeline_module, "get_embedding_service", lambda: StructuredFakeEmbeddingService())
    memory_client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: memory_client)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _setup_product_fully_processed(client, slug_suffix: str):
    """Runs the whole pipeline up through sentiment analysis, stopping right
    before the LLM stage — mirrors how this would actually be used."""
    r = client.post("/companies", json={"name": "InsightCo", "slug": f"insightco-{slug_suffix}"})
    company_id = r.json()["id"]
    r = client.post(f"/companies/{company_id}/products", json={"name": "Phone", "slug": f"phone-{slug_suffix}"})
    product_id = r.json()["id"]

    reviews_payload = [
        {"source": "amazon", "source_review_id": f"i{slug_suffix}{i}", "raw_text": text}
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
    return company_id, product_id


def test_label_topics_writes_back_correctly(client, monkeypatch):
    company_id, product_id = _setup_product_fully_processed(client, "labeling")

    def stub_generate_json(system_prompt, user_prompt, **kwargs):
        return fake_labeling_response(user_prompt)

    monkeypatch.setattr(insights_module, "generate_json", stub_generate_json)

    r = client.post(f"/companies/{company_id}/products/{product_id}/insights/label-topics")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["topics_labeled"] > 0

    topics = client.get(f"/companies/{company_id}/products/{product_id}/topics").json()
    for t in topics:
        assert t["ai_summary"] is not None
        assert t["description"] is not None
        # These are JSON-encoded lists in the DB; confirm they parse back correctly
        complaints = json.loads(t["common_complaints"])
        compliments = json.loads(t["common_compliments"])
        assert isinstance(complaints, list)
        assert isinstance(compliments, list)


def test_executive_summary_requires_labeling_first(client):
    company_id, product_id = _setup_product_fully_processed(client, "execorder")

    # Try executive summary BEFORE labeling — should be skipped, not crash
    r = client.post(f"/companies/{company_id}/products/{product_id}/insights/executive-summary")
    assert r.json()["status"] == "skipped"


def test_executive_summary_uses_topic_summaries(client, monkeypatch):
    company_id, product_id = _setup_product_fully_processed(client, "execsummary")

    def stub_generate_json(system_prompt, user_prompt, **kwargs):
        if "cluster_label" in user_prompt:
            return fake_labeling_response(user_prompt)
        # Executive summary call — verify it received topic-level data, not raw reviews
        topic_summaries = json.loads(user_prompt)
        assert all("summary" in t for t in topic_summaries)
        assert all("raw_text" not in json.dumps(t) for t in topic_summaries)
        return {"executive_summary": "Overall sentiment is mixed across battery, camera, and screen."}

    monkeypatch.setattr(insights_module, "generate_json", stub_generate_json)

    client.post(f"/companies/{company_id}/products/{product_id}/insights/label-topics")
    r = client.post(f"/companies/{company_id}/products/{product_id}/insights/executive-summary")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert "mixed" in body["executive_summary"]

    product = client.get(f"/companies/{company_id}/products/{product_id}").json()
    assert product["executive_summary"] == body["executive_summary"]
    assert product["executive_summary_updated_at"] is not None


def test_malformed_llm_response_does_not_crash_pipeline(client, monkeypatch):
    company_id, product_id = _setup_product_fully_processed(client, "malformed")

    def stub_generate_json_raises(system_prompt, user_prompt, **kwargs):
        raise LLMGenerationError("LLM response was not valid JSON: mock failure")

    monkeypatch.setattr(insights_module, "generate_json", stub_generate_json_raises)

    r = client.post(f"/companies/{company_id}/products/{product_id}/insights/label-topics")
    assert r.status_code == 200  # the endpoint itself shouldn't 500
    body = r.json()
    assert body["status"] == "error"
    assert "reason" in body


def test_full_generate_endpoint_sequences_both_calls(client, monkeypatch):
    company_id, product_id = _setup_product_fully_processed(client, "fullgen")

    def stub_generate_json(system_prompt, user_prompt, **kwargs):
        if "cluster_label" in user_prompt:
            return fake_labeling_response(user_prompt)
        return {"executive_summary": "Test summary."}

    monkeypatch.setattr(insights_module, "generate_json", stub_generate_json)

    r = client.post(f"/companies/{company_id}/products/{product_id}/insights/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["labeling"]["status"] == "completed"
    assert body["executive_summary"]["status"] == "completed"
