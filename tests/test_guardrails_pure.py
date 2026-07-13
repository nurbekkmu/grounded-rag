"""Pure-function guardrails tests — the NLI-model path has its own
offline check (python guardrails.py --selftest), kept out of unit tests
to avoid a model download in CI."""

from guardrails import evidence_too_weak, normalize_claim


def test_attribution_phrases_are_stripped():
    s = normalize_claim("According to the book, RRF sums reciprocal ranks.")
    assert s == "RRF sums reciprocal ranks."
    s = normalize_claim("Two solutions named in the book are X and Y , .")
    assert "in the book" not in s
    assert s.endswith("X and Y.")


def test_normalize_keeps_ordinary_sentences():
    s = "Chunking strategy significantly impacts retrieval performance."
    assert normalize_claim(s) == s


def test_evidence_floor():
    weak = [{"rerank_score": -9.4}, {"rerank_score": -11.0}]
    strong = [{"rerank_score": 4.2}, {"rerank_score": -2.0}]
    assert evidence_too_weak(weak, 0.0)
    assert not evidence_too_weak(strong, 0.0)
    assert evidence_too_weak([], 0.0)
