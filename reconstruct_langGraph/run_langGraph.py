from langgraph.graph import StateGraph, START, END

from langGraph_node import GraphState
# from reconstruct_langGraph.langGraph_evaluation import evaluate_node
from langGraph_reader import load_test_data_node, predict_reader_node, predict_node
from langGraph_retriever import retrieval_pipeline_node


# 构建图 (Workflow) 包括retrival、load_data、predict和evaluate节点，并定义它们之间的依赖关系
workflow = StateGraph(GraphState)

# workflow.add_node("retrieval", retrieval_pipeline_node)
# workflow.add_node("load_data", load_test_data_node)
# workflow.add_node("predict and eval", predict_node)
# # workflow.add_node("evaluate", evaluate_node)
#
# workflow.add_edge(START, "retrieval")
# workflow.add_edge("retrieval", "load_data")
# workflow.add_edge("load_data", "predict and eval")
# workflow.add_edge("predict and eval", END)


# 考虑到数据量大，从reader开始
workflow.add_node("predict and eval", predict_reader_node)
# workflow.add_node("evaluate", evaluate_node)

workflow.add_edge(START, "predict and eval")
workflow.add_edge("predict and eval", END)

app = workflow.compile()

# 运行配置
if __name__ == "__main__":
    initial_state = {
        "wiki_dump_path": "data/enwiki-latest-pages-articles.xml.bz2",
        "grouped_dir": "data/grouped_results",
        "test_data_name": "nq",
        "test_data_split": "subset_100",
        # "output_file_path": "./replicate_exp_langG/nq_gpt4o_100.json",
        # "reader_model": "GPT-4o",
        "output_file_path": "./replicate_exp_langG/nq_claude_4.6_sonnet.json",
        "reader_model": "Claude-sonnet-4-6",
        # "output_file_path": "./replicate_exp_langG/nq_gpt-5.4-pro_100.json",
        # "reader_model": "GPT-5.4-pro",
        "args": { # retrival和reader共用的参数
            "test_data_path": "./data/nq_test.json",
            "test_data_name": "nq",
            "top_k": 1
        }
    }
    app.invoke(initial_state)