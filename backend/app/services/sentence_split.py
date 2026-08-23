import re

# Splits on '.', '!', '?' followed by whitespace or end-of-string. Deliberately
# simple — a real sentence tokenizer (e.g. nltk's punkt) handles abbreviations
# ("Mr.", "e.g.") better, but punkt requires a data download we can't rely on
# being available (same class of limitation as the huggingface.co restriction
# noted for embeddings). Product reviews are short and rarely hit those edge
# cases badly enough to matter for aspect-matching purposes.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]
