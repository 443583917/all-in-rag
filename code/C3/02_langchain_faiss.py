from langchain_community.vectorstores import FAISS #一个高效的相似度搜索库，用于存储和检索向量。
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
#Document(page_content="张三是法外狂徒", metadata={}), metadata 没有传值构建
docs = [Document(page_content=t) for t in texts]
# 为什么要转换为doc格式？
# 1.LangChain 的很多模块（比如 FAISS向量存储、检索器、RAG）都要求输入是 Document 类型，而不是纯字符串。
# 2.这样不仅能保存文本，还能保存来源信息（metadata），方便后续追溯。
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
vectorstore = FAISS.from_documents(docs, embeddings)
# 这边会将docs序列化成index.faiss和index.pkl两个文件
# FAISS 的二进制索引文件，存储向量和索引结构
# 用 pickle 序列化的 Python 对象包含 文档对象（Document，含 page_content 和 metadata）。
# 索引的配置参数（比如维度、embedding 模型信息）。
# 文档 ID 与向量的映射关系。
local_faiss_path = r"D:\GitHub\all-in-rag\code\C3"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
# 额外小知识 LangChain 默认是 禁止反序列化，避免安全隐患。
# 如果确保需要加载的数据是安全的就可以反序列化把二进制文件重新还原成 Python 对象
)
# loaded_vectorstore 就是一个 已经包含了文档向量和索引的数据库对象
# 向量只是文本的坐标映射不能直接转换成对应的文本
# loaded_vectorstore用文本对应的向量作为索引提高查询输出生产中常用
# 执行相似性搜索
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")