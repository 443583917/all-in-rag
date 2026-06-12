import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# 1. 初始化模型与嵌入
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 可换成 gpt-4-turbo 等
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# 2. 定义状态结构
class RetrievalState:
    query: str
    documents: list
    sentence_index: VectorStoreIndex
    base_index: VectorStoreIndex
    sentence_result: str
    base_result: str
    summary: str

# 3. 定义节点函数
def load_docs(state: RetrievalState):
    state.documents = SimpleDirectoryReader(
        input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
    ).load_data()
    return state

def build_indexes(state: RetrievalState):
    # 句子窗口索引
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
    sentence_nodes = node_parser.get_nodes_from_documents(state.documents)
    state.sentence_index = VectorStoreIndex(sentence_nodes)

    # 常规分块索引
    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(state.documents)
    state.base_index = VectorStoreIndex(base_nodes)
    return state

def query_indexes(state: RetrievalState):
    sentence_engine = state.sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
    )
    base_engine = state.base_index.as_query_engine(similarity_top_k=2)

    state.sentence_result = sentence_engine.query(state.query)
    state.base_result = base_engine.query(state.query)
    return state

def summarize_results(state: RetrievalState):
    prompt = f"""
    用户查询: {state.query}

    句子窗口检索结果:
    {state.sentence_result}

    常规分块检索结果:
    {state.base_result}

    请总结两种检索方式的差异，指出哪种更准确、更连贯，并给出简要分析。
    """
    response = llm.invoke(prompt)
    state.summary = response.content
    return state

def show_results(state: RetrievalState):
    print(f"查询: {state.query}\n")
    print("--- 句子窗口检索结果 ---")
    print(state.sentence_result)
    print("\n--- 常规检索结果 ---")
    print(state.base_result)
    print("\n--- LLM 总结 ---")
    print(state.summary)
    return state

# 4. 构建 LangGraph 工作流
graph = StateGraph(RetrievalState)
graph.add_node("load_docs", load_docs)
graph.add_node("build_indexes", build_indexes)
graph.add_node("query_indexes", query_indexes)
graph.add_node("summarize_results", summarize_results)
graph.add_node("show_results", show_results)

graph.add_edge("load_docs", "build_indexes")
graph.add_edge("build_indexes", "query_indexes")
graph.add_edge("query_indexes", "summarize_results")
graph.add_edge("summarize_results", "show_results")
graph.add_edge("show_results", END)

# 5. 执行
app = graph.compile()
app.invoke({"query": "What are the concerns surrounding the AMOC?"})
