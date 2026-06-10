from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
import requests
import math

# Embedding 一次性处理的数据上限32条
# 必须自定义TEIEmbeddings方法,embeddings方法默认传递
#{
#  "model": "Qwen/Qwen3-Embedding-0.6B",
#  "input": ["文本1", "文本2", ...],
#  "encoding_format": "float"  
#  （Qwen/Qwen3-Embedding-0.6B推理服务器）不支持 encoding_format 字段  
#}

# ========== 自定义 TEI Embedding（支持批处理） ==========
class TEIEmbeddings(Embeddings):
    """兼容 text-embeddings-inference 的 Embedding 类（支持批处理）"""

    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B",
                 url="http://127.0.0.1:8080/v1/embeddings",
                 batch_size=32):
        self.model = model
        self.url = url
        self.batch_size = batch_size  # TEI 最大 batch = 32

    def _embed_batch(self, texts):
        payload = {
            "model": self.model,
            "input": texts
        }
        r = requests.post(self.url, json=payload)
        data = r.json()

        if "data" not in data:
            raise ValueError(f"TEI 返回异常: {data}")

        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        """自动分批，避免 batch size > 32"""
        all_embeddings = []
        total = len(texts)
        batches = math.ceil(total / self.batch_size)

        for i in range(batches):
            batch = texts[i*self.batch_size : (i+1)*self.batch_size]
            print(f"处理 batch {i+1}/{batches}, 大小={len(batch)}")
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text):
        return self._embed_batch([text])[0]


# ========== 1. 定义 State ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


# ========== 2. 文档加载 + 切分 ==========
markdown_path = r"D:\GitHub\all-in-rag\data\C1\markdown\easy-rl-chapter1.md"
loader = UnstructuredLoader(markdown_path)
docs = loader.load()

print(f"文档数量: {len(docs)}")
print("文档示例:", docs[0].page_content[:200])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print(f"切分后的 chunk 数量: {len(chunks)}")
print("第一个 chunk:", chunks[0].page_content[:200])


# ========== 3. 使用 TEI Embedding（容器版 + 批处理） ==========
embeddings = TEIEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",
    url="http://127.0.0.1:8080/v1/embeddings",
    batch_size=32  # TEI 最大 batch 限制
)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)


# ========== 4. 相似搜索测试 ==========
test_query = "强化学习的例子"
results = vectorstore.similarity_search(test_query, k=2)

for i, doc in enumerate(results):
    print(f"检索结果 {i+1}:", doc.page_content[:200])


# ========== 5. 提示词模板 ==========
prompt = ChatPromptTemplate.from_template("""
请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:
""")


# ========== 6. LLM ==========
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=4096,
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)


# ========== 7. Graph 节点 ==========
def retrieve_node(state: State):
    question = state["messages"][-1].content
    retrieved_docs = vectorstore.similarity_search(question, k=5)

    if not retrieved_docs:
        docs_content = "未找到相关上下文"
    else:
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {"context": docs_content, "messages": state["messages"]}


def answer_node(state: State):
    question = state["messages"][-1].content
    context = state["context"]
    response = llm.invoke(prompt.format(question=question, context=context))
    return {"messages": state["messages"] + [response]}


# ========== 8. 构建 Graph ==========
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()


# ========== 9. 测试 ==========
user_input = "文中举了哪些例子？"
result = app.invoke({"messages": [HumanMessage(content=user_input)]})

for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")
