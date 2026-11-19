"""Evaluation subpackage: hand-built corpus and retrieval metrics."""

from .corpus import CATEGORIES, MEMORIES, PERSONAS, QUERIES, Query, build
from .metrics import EvalResult, evaluate, format_report, query_ids

__all__ = [
    "CATEGORIES",
    "MEMORIES",
    "PERSONAS",
    "QUERIES",
    "Query",
    "EvalResult",
    "build",
    "evaluate",
    "format_report",
    "query_ids",
]
