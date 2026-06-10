from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ========== 1. 定义 State ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

# ========== 2. 文档加载 + 切分 + 向量库 ==========

markdown_path = r"D:\GitHub\all-in-rag\data\C1\markdown\easy-rl-chapter1.md"
loader = UnstructuredLoader(markdown_path)
docs = loader.load()
print(f"文档数量: {len(docs)}")
print("文档示例:", docs[0].page_content[:200])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"切分后的 chunk 数量: {len(chunks)}")
print("第一个 chunk:", chunks[0].page_content[:200])
# 保存然后查看内容
with open("all_chunks.txt", "w", encoding="utf-8") as f:
    for i, doc in enumerate(chunks, start=1):
        f.write(f"chunk={i} 内容:\n")
        f.write(doc.page_content.strip() + "\n\n")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)
# ========== 相似的搜索 ==========
test_query = "强化学习的例子"
results = vectorstore.similarity_search(test_query, k=2)
for i, doc in enumerate(results):
    print(f"检索结果 {i+1}:", doc.page_content[:200])
# ========== 3. 提示词模板 ==========
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

# ========== 4. LLM ==========
llm = ChatOpenAI(
    temperature=0.7,# 创造性
    max_tokens=4096,# 最大输出 越小越快
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)

# ========== 5. 定义 Graph 节点 ==========
def retrieve_node(state: State):
    # 从向量库检索
    question = state["messages"][-1].content
    retrieved_docs = vectorstore.similarity_search(question, k=5)  # 调大 k 值
    if not retrieved_docs:
        print("未检索到相关文档")
        docs_content = "未找到相关上下文"
    else:
        # 打印前几个 chunk，方便调试
        print(" 检索到的文档内容示例:")
        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"--- 文档 {i+1} ---")
            print(doc.page_content[:300])  # 打印前 300 字
            print("元信息:", doc.metadata)
            print("------------------")
            # 将检索结果写入 txt 文件
        with open("retrieved_chunks.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(retrieved_docs, start=1):
                f.write(f"k={i} 内容:\n")
                f.write(doc.page_content + "\n\n")
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {"context": docs_content, "messages": state["messages"]}

def answer_node(state: State):
    question = state["messages"][-1].content
    context = state["context"]
    response = llm.invoke(prompt.format(question=question, context=context))
    return {"messages": state["messages"] + [response]}

# ========== 6. 构建 Graph ==========
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# ========== 7. 测试 ==========
user_input = "文中举了哪些例子？"
result = app.invoke({"messages": [HumanMessage(content=user_input)]})

for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")
