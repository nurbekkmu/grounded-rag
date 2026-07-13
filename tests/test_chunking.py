from chunk_blog import blocks, chunk_doc, merge_tiny


def test_blocks_split_on_headings_and_blank_lines():
    md = "intro para\n\n## Section A\n\npara one\n\npara two"
    out = blocks(md)
    assert out[0] == (None, "intro para")
    assert out[1] == ("Section A", "para one")
    assert out[2] == ("Section A", "para two")


def test_code_fences_stay_atomic_even_with_hash_lines():
    md = "## Code\n\n```python\n# not a heading\n\nx = 1\n```\n\nafter"
    out = blocks(md)
    fence = next(t for _, t in out if t.startswith("```"))
    assert "# not a heading" in fence and "x = 1" in fence
    assert all(h != "not a heading" for h, _ in out)


def test_merge_tiny_folds_fragment_forward():
    chunks = [
        {"chunk_id": "blog/x#000", "n_tokens": 20, "text": "tiny intro"},
        {"chunk_id": "blog/x#001", "n_tokens": 500, "text": "big " * 400},
    ]
    merged = merge_tiny(chunks, "blog/x")
    assert len(merged) == 1
    assert merged[0]["text"].startswith("tiny intro")
    assert merged[0]["chunk_id"] == "blog/x#000"


def test_chunk_doc_emits_uniform_schema():
    doc = {"doc_id": "blog/t", "title": "T", "url": "u", "date": "d",
           "text": "## S\n\n" + ("word " * 300).strip()}
    chunks = chunk_doc(doc)
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    for field in ("chunk_id", "doc_id", "source", "section_path",
                  "url", "date", "n_tokens", "text"):
        assert field in c
    assert c["source"] == "blog"
    assert c["chunk_id"] == "blog/t#000"
