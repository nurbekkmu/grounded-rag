from contextualize import cache_key, load_prompt, split_template
from generate import CITE_RE


def test_cache_key_stable_and_sensitive():
    k1 = cache_key("doc", "chunk", 1, "model-a")
    assert k1 == cache_key("doc", "chunk", 1, "model-a")
    assert k1 != cache_key("doc", "chunk", 2, "model-a")   # prompt bump
    assert k1 != cache_key("doc", "chunk", 1, "model-b")   # model swap
    assert k1 != cache_key("doc2", "chunk", 1, "model-a")  # rechunk


def test_split_template_reassembles_to_original_shape():
    template = ("<document>\n{{WHOLE_DOCUMENT}}\n</document>\n"
                "Here is the chunk:\n<chunk>\n{{CHUNK_CONTENT}}\n</chunk>")
    head, tail = split_template(template)
    assert head.endswith("</document>")
    assert "{{WHOLE_DOCUMENT}}" in head
    assert tail.startswith("Here is the chunk")
    assert "{{CHUNK_CONTENT}}" in tail


def test_shipped_prompts_are_splittable_and_versioned():
    for path in ("prompts/contextualize.yaml",
                 "prompts/contextualize_lite.yaml"):
        cfg = load_prompt(path)
        head, tail = split_template(cfg["template"])
        assert head and tail
        assert isinstance(cfg["version"], int)


def test_citation_regex_matches_ids_not_prose():
    text = "Real [book/ch06#014] and [blog/agents#009], not [12] or [foo]."
    assert CITE_RE.findall(text) == ["book/ch06#014", "blog/agents#009"]
