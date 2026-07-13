"""
Runtime configuration: config.yaml (repo root) overrides built-in
defaults; CLI flags override both. Scripts use cfg() for their argparse
defaults, so `python src/retrieve.py "q"` obeys config.yaml while
`--mode bm25` still wins.

Build-time tunables (chunk sizes, embedding model) are deliberately not
configurable here — changing them requires rebuilding artifacts, and
their homes document that chain.
"""

import os

import yaml

_DEFAULTS = {
    "retrieval.mode": "hybrid",
    "retrieval.top_n": 75,
    "retrieval.index": "data/index/baseline",
    "rerank.model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank.keep": 8,
    "guardrails.min_rerank": 0.0,
    "generation.prompt": "prompts/answer_with_citations.yaml",
}


def _load_file():
    if os.path.exists("config.yaml"):
        with open("config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_FILE = _load_file()


def cfg(path: str, default=None):
    """cfg("rerank.keep") -> config.yaml value, else built-in default."""
    node = _FILE
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _DEFAULTS.get(path, default)
        node = node[part]
    return node
