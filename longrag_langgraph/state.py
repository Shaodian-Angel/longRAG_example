from __future__ import annotations

from typing import Any, Optional, TypedDict

from rag_contracts import GenerationResult, RetrievalResult


class LongRAGGraphState(TypedDict, total=False):
    # Input
    query: str
    query_id: str
    answers: list[str]
    test_data_name: str  # "nq" or "hotpotqa"

    # Stage outputs
    expanded_queries: list[str]
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult

    # Error tracking
    error: Optional[str]
