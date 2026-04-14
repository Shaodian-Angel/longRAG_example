from tqdm import tqdm
import tiktoken
import subprocess
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langGraph_node import GraphState
from utils.load_data_util import load_json_file, load_retrieval_txt


# 保留原工作流
def predict_reader_node(state: GraphState):
    """节点：使用 subprocess 运行原有的 eval_qa.py"""
    print(f"--- 正在运行 Reader 预测: {state['reader_model']} ---")

    # 构造命令行参数，对应 run_eval_qa.sh 中的逻辑
    cmd = [
        "python", "eval/eval_qa.py",
        "--test_data_name", state["test_data_name"],
        "--test_data_split", state["test_data_split"],
        "--output_file_path", state["output_file_path"],
        "--reader_model", state["reader_model"]
    ]

    try:
        # 运行原有的评估代码
        # 如果需要解决代理问题，可以在这里传入 env 参数
        env = os.environ.copy()
        # env["HTTP_PROXY"] = "..."

        subprocess.run(cmd, check=True, env=env)
        return {"prediction_done": True}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# 重写方法
def load_test_data_node(state: GraphState):
    """加载测试集"""
    print("--- 加载测试数据 ---")
    data = load_json_file(state["args"]["test_data_path"])
    return {"test_data": data}


def predict_node(state: GraphState):
    """
    整合 Reader 逻辑：连接检索排名和语料内容
    """
    print("--- 正在运行 Reader (GPT-4o) ---")

    # 1. 读取 Tevatron 排名结果
    top_k = state["args"].get("top_k", 1)
    ranking_df = load_retrieval_txt(state["ranking_file"], n_retrieve=top_k)
    # 转换为字典: {query_id: [doc_id1, doc_id2, ...]}
    ranking_dict = ranking_df.groupby('q_id')['doc_id'].apply(list).to_dict()

    # 2. 初始化 Reader
    llm_inference = GPTInference()
    results = []
    enc = tiktoken.get_encoding("cl100k_base")

    for item in tqdm(state["test_data"], desc="Predicting"):
        q_id = str(item["query_id"])
        question = item["query"]
        answers = item["answer"]

        # 3. 根据检索 ID 拼接 Context
        retrieved_ids = ranking_dict.get(q_id, [])
        # 从 corpus_index 中找回文本
        context_list = [state["corpus_index"].get(str(d_id), "") for d_id in retrieved_ids]
        full_context = "\n\n".join(context_list)

        # 4. 调用 Reader 模型
        try:
            if state["args"]["test_data_name"] == "nq":
                long_ans, short_ans = llm_inference.predict_nq(full_context, question, ["Retrieved Docs"])
            else:
                long_ans, short_ans = llm_inference.predict_hotpotqa(full_context, question, ["Retrieved Docs"])
        except Exception as e:
            print(f"Error at {q_id}: {e}")
            long_ans, short_ans = "", ""

        results.append({
            "query_id": q_id,
            "question": question,
            "answers": answers,
            "long_ans": long_ans,
            "short_ans": short_ans,
            "context_size": len(enc.encode(full_context))
        })

    return {"predictions": results}

