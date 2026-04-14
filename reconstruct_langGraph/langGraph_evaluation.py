from langGraph_node import GraphState
from utils.eval_util import single_ans_em


def evaluate_node(state: GraphState):
    """评估 EM 分数"""
    print("--- 评估结果 ---")
    exact_match = 0
    for res in state["predictions"]:
        exact_match += single_ans_em(res["short_ans"], res["answers"])

    em_score = exact_match / len(state["predictions"]) if state["predictions"] else 0
    return {"evaluation_results": {"exact_match": em_score}}