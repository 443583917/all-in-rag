from llama_index.core import VectorStoreIndex, Document, Settings
from langchain_huggingface import HuggingFaceEmbedding

# LlamaIndex 框架 
# 专注于“数据接入、索引和检索”的框架，用来构建上下文增强型 LLM 应用（尤其是 RAG 系统）
# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = r"D:\GitHub\all-in-rag\code\C3\llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")
