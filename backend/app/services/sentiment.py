"""
Traditional lexicon-based sentiment scoring (VADER), deliberately NOT an LLM
call — this runs per-sentence, potentially millions of times at scale, and
the spec is explicit that the LLM must not be called per review.

VADER is tuned for short, informal text (social media, reviews) which is
exactly what we're scoring here, and its lexicon ships with the package —
no network access needed at runtime.
"""

from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@lru_cache(maxsize=1)
def _get_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> tuple[str, float]:
    """Returns (label, compound_score) where label is 'positive'/'negative'/'neutral'
    and compound_score is in [-1.0, 1.0]. Thresholds match VADER's own documented
    convention (compound >= 0.05 -> positive, <= -0.05 -> negative)."""
    if not text or not text.strip():
        return "neutral", 0.0

    scores = _get_analyzer().polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return label, compound
