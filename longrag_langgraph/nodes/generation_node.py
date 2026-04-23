from __future__ import annotations

from rag_contracts import Generation


def build_node(generation: Generation):
    """Build a generation node that accepts canonical Generation via DI."""

    async def node(state):
        query = state["query"]
        context = state.get("retrieval_results", [])
        test_data_name = state.get("test_data_name", "nq")
        instruction = f"dataset={test_data_name}"

        result = generation.generate(
            query=query, context=context, instruction=instruction
        )
        return {"generation_result": result}

    return node
