from index import bm25_terms, collection_for
from retrieve import rrf_fuse


def test_rrf_orders_by_summed_reciprocal_rank():
    fused = rrf_fuse({"a": ["x", "y"], "b": ["y", "z"]})
    order = [c["chunk_id"] for c in fused]
    assert order == ["y", "x", "z"]          # y appears in both lists
    assert fused[0]["ranks"] == {"a": 2, "b": 1}


def test_rrf_single_arm_passthrough_order():
    fused = rrf_fuse({"bm25": ["a", "b", "c"]})
    assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]


def test_bm25_terms_preserve_identifiers():
    terms = bm25_terms("Use text-embedding-3-small, GPT-4 and nomic v1.5!")
    assert "text-embedding-3-small" in terms
    assert "gpt-4" in terms
    assert "v1.5" in terms


def test_bm25_terms_lowercase_and_split():
    assert bm25_terms("BM25 beats TF-IDF?") == ["bm25", "beats", "tf-idf"]


def test_collection_names_are_distinct_per_index_dir():
    a = collection_for("data/index/baseline")
    b = collection_for("data/index/book-contextual")
    assert a != b
    assert a.startswith("Chunk_") and b.startswith("Chunk_")
