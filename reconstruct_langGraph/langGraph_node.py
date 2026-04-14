from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from tqdm import tqdm

from utils.gpt_inference import GPTInference
from utils.load_data_util import load_json_file


# 1. 定义状态 (State)
# 这是在节点之间传递的对象
class GraphState(TypedDict):
    test_data: List[dict]
    predictions: List[dict]
    context_sizes: List[int]
    evaluation_results: Dict[str, float]
    args: dict
    test_data_path: str


# 2. 定义节点逻辑 (Nodes)
# 这里的函数接收当前 state，并返回要更新到 state 中的键值对

# Retriever
def load_data_node(state: GraphState):
    print("---LOADING DATA---")
    # 假设 load_json_file 已在外部导入
    data = load_json_file(state["test_data_path"])
    return {"test_data": data}


# Reader
def predict_node(state: GraphState):
    print("---PREDICTING---")
    llm_inference = GPTInference()
    test_data = state["test_data"]
    args = state["args"]

    results = []
    context_sizes = []
    for item in tqdm(test_data, desc="Evaluating QA"):
        # ... 你的预测逻辑 ...
        # (保持原有的 try-except 逻辑)
        results.append({"query_id": item["query_id"], "short_ans": "example"})  # 示例

    return {"predictions": results, "context_sizes": context_sizes}


# Evaluator
def evaluate_node(state: GraphState):
    print("---EVALUATING---")
    # ... 你的评估逻辑 ...
    return {"evaluation_results": {"exact_match": 0.85}}  # 示例


# 保存结果节点
def save_results_node(state: GraphState):
    print("---SAVING---")
    # ... 你的保存逻辑 ...
    return state


# 3. 构建图 (Build the Graph)
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("load_data", load_data_node)
workflow.add_node("predict", predict_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("save_results", save_results_node)

# 设置边 (Edges)
workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "predict")
workflow.add_edge("predict", "evaluate")
workflow.add_edge("evaluate", "save_results")
workflow.add_edge("save_results", END)

# 编译
app = workflow.compile()

# 4. 运行
initial_state = {
    "test_data_path": "./data/test_data.json",
    "args": {
        "test_data_name": "nq",
        "output_file_path": "./replicate_exp/nq_gpt4o_100.json"
    }
}

final_output = app.invoke(initial_state)
print(final_output["evaluation_results"])