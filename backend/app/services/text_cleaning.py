import re

from langdetect import LangDetectException, detect

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(raw_text: str) -> str:
    """Basic normalization: strip URLs/HTML/emoji, collapse whitespace.

    Deliberately conservative — we do NOT lowercase or strip punctuation here,
    because downstream sentiment/embedding models perform better on natural text.
    Aggressive normalization (stemming, stopword removal) is left to the
    keyword-extraction step only, not applied globally.
    """
    text = raw_text.strip()
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def detect_language(text: str) -> str | None:
    """Returns an ISO 639-1 code (e.g. 'en') or None if detection fails
    (common for very short strings like 'Good!')."""
    if not text or len(text) < 3:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
