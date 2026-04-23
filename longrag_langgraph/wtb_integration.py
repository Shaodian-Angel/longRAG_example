"""WTB integration for the LongRAG LangGraph pipeline.

Provides a zero-arg ``graph_factory`` suitable for ``WorkflowProject``,
plus helper functions for registering canonical component variants.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from rag_contracts import Generation, Query, Reranking, Retrieval

from .main_pipeline import build_graph


def create_longrag_graph_factory(
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
) -> Callable:
    """Return a zero-arg factory that builds the LongRAG LangGraph.

    All component arguments are captured in the closure so that the
    returned callable matches ``WorkflowProject.graph_factory`` signature.
    """

    def factory():
        return build_graph(
            retrieval=retrieval,
            generation=generation,
            reranking=reranking,
            query=query,
        )

    return factory


def create_longrag_project(
    name: str = "longrag_langgraph",
    *,
    retrieval: Retrieval,
    generation: Generation,
    reranking: Reranking | None = None,
    query: Query | None = None,
) -> Any:
    """Create a ``WorkflowProject`` for the LongRAG pipeline.

    Returns the project so callers can further ``register_variant()`` on it.
    """
    from wtb.sdk import WorkflowProject

    factory = create_longrag_graph_factory(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query,
    )

    project = WorkflowProject(
        name=name,
        graph_factory=factory,
        description="LongRAG QA pipeline (LangGraph) -- long retrieval units + long-context reader",
    )
    return project
