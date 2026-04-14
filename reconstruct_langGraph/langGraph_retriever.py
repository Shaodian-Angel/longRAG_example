import subprocess
import os

from langGraph_node import GraphState
from utils.load_data_util import load_dict_pickle


def wiki_preprocess_node(state: GraphState):
    """
    节点 1: 对应 extract_and_clean_wiki_dump.sh
    执行 WikiExtractor 清洗数据
    """
    print("--- 步骤 1: 提取并清洗 Wiki Dump ---")
    input_file = state["wiki_dump_path"]
    # 调用脚本 (假设脚本在 scripts 目录下)
    subprocess.run(["sh", "scripts/extract_and_clean_wiki_dump.sh", input_file], check=True)

    output_txt = input_file.split('.')[0] + ".txt"
    return {"cleaned_text_path": output_txt}


def group_docs_node(state: GraphState):
    """
    节点 2: 对应 process_wiki_page.sh 和 group_documents.sh
    将清洗后的文本处理并按 LongRAG 逻辑分组
    """
    print("--- 步骤 2: 文档分组 (LongRAG Core) ---")
    # 1. 运行 process_wiki_page
    subprocess.run(["sh", "scripts/process_wiki_page.sh"], check=True)
    # 2. 运行 group_documents
    subprocess.run(["sh", "scripts/group_documents.sh"], check=True)

    return {"grouped_dir": "data/grouped_results"}  # 对应脚本中的 output_dir


def tevatron_retrieve_node(state: GraphState):
    """
    节点 3: 对应 run_retrieve_tevatron.sh
    多 GPU 编码并进行语义搜索
    """
    print("--- 步骤 3: 向量化检索 (Tevatron) ---")
    # 针对你之前的代理问题，如果下载模型需要代理，可以在这里设置
    my_env = os.environ.copy()
    # my_env["HTTP_PROXY"] = "http://127.0.0.1:xxxx"

    subprocess.run(["sh", "scripts/run_retrieve_tevatron.sh"], env=my_env, check=True)

    return {"ranking_file": "hqa_rank_200_new.txt"}


# Summary
def retrieval_pipeline_node(state: GraphState):
    """
    整合节点：执行从清洗到检索的所有 .sh 脚本
    """
    print("--- 正在运行 Retrieval 流水线 (Wiki Clean -> Grouping -> Tevatron) ---")

    # 步骤 1: 清洗 Wiki Dump
    subprocess.run(["wsl","sh", "scripts/extract_and_clean_wiki_dump.sh", state["wiki_dump_path"]], check=True)
    # 步骤 2: 处理并分组文档
    subprocess.run(["wsl","sh", "scripts/process_wiki_page.sh"], check=True)
    subprocess.run(["wsl","sh", "scripts/group_documents.sh"], check=True)

    # 步骤 3: 运行 Tevatron 检索
    subprocess.run(["wsl","sh", "scripts/run_retrieve_tevatron.sh"], check=True)

    # 加载分组后的语料库索引 (假设 group_documents 输出为 pickle 或 jsonl)
    # 这里使用你的 load_dict_pickle 或 load_json_file
    print("--- 正在构建语料库索引 ---")
    # 假设输出路径为 state["grouped_dir"] 下的 long_corpus.pkl
    corpus_index = load_dict_pickle(os.path.join(state["grouped_dir"], "long_corpus.pkl"))

    return {"corpus_index": corpus_index, "ranking_file": "hqa_rank_200_new.txt"}