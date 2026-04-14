from typing import TypedDict, List, Dict, Optional


# 定义图状态 (State)
class GraphState(TypedDict):
    # 路径与配置(retrieval)
    wiki_dump_path: str
    grouped_dir: str
    ranking_file: str

    # 测试集与模型配置(read)
    test_data_name: str  # "nq" 或 "hotpotqa"
    test_data_split: str  # "subset_100" 等
    output_file_path: str
    reader_model: str  # "GPT-4o", "Gemini", "Claude"

    # 状态追踪
    retrieval_done: bool
    prediction_done: bool
    error: Optional[str]

    # 数据流
    test_data: List[dict]
    corpus_index: Dict[str, str]  # doc_id -> text 映射
    predictions: List[dict]

    # 结果与配置
    evaluation_results: Dict[str, float]
    args: dict