import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from typing import TypedDict

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================
# 0. 定义 LangGraph 状态 (State)
# ============================================================
# State 是图中各节点之间传递数据的"共享内存"
# 每个节点读取 state 中的字段，处理后返回更新后的字段
class RAGState(TypedDict):
    question: str       # 用户问题
    context: str        # 检索到的上下文
    answer: str         # LLM 生成的回答


# ============================================================
# 1. 准备文档 & 向量库（与 LlamaIndex 的 VectorStoreIndex 对应）
# ============================================================
markdown_path = "../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地 markdown 文件
loader = UnstructuredLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "]
)
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")

# 配置大语言模型
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=4096,
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)


# ============================================================
# 2. 定义 LangGraph 节点 (Nodes)
# ============================================================
# 每个节点是一个普通的 Python 函数，接收 state，返回 state 的部分更新

def retrieve_node(state: RAGState) -> dict:
    """检索节点：把用户问题转为向量，在向量库中搜索相关文档片段"""
    question = state["question"]
    # 检索 top 3 最相似的 chunk
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    # 拼接为上下文字符串
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return {"context": context}


def generate_node(state: RAGState) -> dict:
    """生成节点：将上下文 + 问题填入提示词，调用 LLM 生成回答"""
    formatted_prompt = prompt.format(
        question=state["question"],
        context=state["context"]
    )
    response = llm.invoke(formatted_prompt)
    return {"answer": response.content}


# ============================================================
# 3. 构建 StateGraph（对应 LlamaIndex 的 index → query_engine 流程）
# ============================================================
# StateGraph 是 LangGraph 的核心：
#   - 用 state 定义节点间共享的数据结构
#   - 用 add_node 注册处理函数
#   - 用 add_edge / add_conditional_edge 定义流程走向

graph = StateGraph(RAGState)

# 注册节点
graph.add_node("retrieve", retrieve_node)   # 检索阶段
graph.add_node("generate", generate_node)   # 生成阶段

# 定义流程边
# LlamaIndex 的 index.as_query_engine().query() 内部也是：
#   用户问题 → 向量检索 → 拼接上下文 → LLM 生成 → 返回答案
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 编译为可执行对象
rag_app = graph.compile()


# ============================================================
# 4. 执行查询
# ============================================================
question = "文中举了哪些例子?"

# LangGraph 调用方式：调用 .invoke(初始state) 自动按图流动
result = rag_app.invoke({"question": question})

# 输出
print("=" * 60)
print(f"问题: {question}")
print("=" * 60)
print(f"回答:\n{result['answer']}")
print("=" * 60)
print(f"检索到的上下文片段:\n{result['context'][:500]}...")
