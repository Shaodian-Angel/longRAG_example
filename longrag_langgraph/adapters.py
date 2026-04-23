"""Adapters between LongRAG internals and canonical rag_contracts.

Forward direction: wrap existing LongRAG classes to satisfy canonical protocols.
Reverse direction: wrap canonical protocols for use by legacy LongRAG code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval adapter -- wraps pre-computed HuggingFace dataset context
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HFDatasetRetrieval:
    """``rag_contracts.Retrieval`` backed by the pre-joined context in the
    HuggingFace ``TIGER-Lab/LongRAG`` dataset.

    Each dataset item already has ``context`` and ``context_titles`` fields,
    so "retrieval" is just a lookup keyed on the query.
    """

    dataset_name: str = "nq"
    dataset_split: str = "test"
    _index: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def _ensure_loaded(self) -> None:
        if self._index:
            return
        from datasets import load_dataset

        ds = load_dataset("TIGER-Lab/LongRAG", self.dataset_name, split=self.dataset_split)
        for item in ds:
            qid = str(item["query_id"])
            self._index[qid] = item
            self._index[item["query"]] = item

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]:
        self._ensure_loaded()
        results: list[RetrievalResult] = []
        for q in queries:
            item = self._index.get(q)
            if item is None:
                continue
            titles = item.get("context_titles", [])
            if isinstance(titles, str):
                titles = [titles]
            results.append(
                RetrievalResult(
                    source_id=str(item.get("query_id", q)),
                    content=item.get("context", ""),
                    score=1.0,
                    title=", ".join(titles) if titles else "",
                    metadata={
                        "context_titles": titles,
                        "query_id": str(item.get("query_id", "")),
                    },
                )
            )
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Generation adapter -- wraps existing LLM inference classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LongRAGGeneration:
    """``rag_contracts.Generation`` wrapping LongRAG's GPT/Claude/Gemini readers.

    The ``instruction`` parameter is expected to contain ``dataset=nq`` or
    ``dataset=hotpotqa`` to select the appropriate prompt template.
    """

    llm_inference: Any

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        full_context = "\n\n".join(r.content for r in context)
        titles = []
        for r in context:
            ctx_titles = r.metadata.get("context_titles", [])
            if ctx_titles:
                titles.extend(ctx_titles)
            elif r.title:
                titles.append(r.title)

        dataset = "nq"
        if "hotpot" in instruction.lower():
            dataset = "hotpotqa"

        try:
            if dataset == "nq":
                long_ans, short_ans = self.llm_inference.predict_nq(
                    full_context, query, titles
                )
            else:
                long_ans, short_ans = self.llm_inference.predict_hotpotqa(
                    full_context, query, titles
                )
        except Exception:
            long_ans, short_ans = "", ""

        return GenerationResult(
            output=short_ans,
            citations=[r.source_id for r in context],
            metadata={
                "long_answer": long_ans,
                "short_answer": short_ans,
                "dataset": dataset,
            },
        )
