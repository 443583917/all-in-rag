import os
os.environ["HF_TOKEN"] = "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"
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# 1. 配置模型（用 LlamaIndex 原生 OpenAI 接口）
Settings.llm = OpenAI(
    temperature=0.1,# 创造性
    max_tokens=4096,# 最大输出 越小越快
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)

# 2. 配置 Embedding（用 LlamaIndex 原生 HuggingFaceEmbedding）
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 3. 加载文档
documents = SimpleDirectoryReader(
    input_files=[r"D:\GitHub\all-in-rag\data\C3\pdf\IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 4. 创建节点与索引
# 4.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,#window_size=3 表示除了当前句子，还会额外带上前后各 3 个句子作为上下文。
    window_metadata_key="window",# 当前句子和上下文拼接后的结果
    original_text_metadata_key="original_text",#存储的是 当前句子本身
)
#假设原文是句子1：机器学习是一门研究如何让计算机从数据中学习的学科。
#句子2：深度学习是机器学习的一个子领域。
#句子3：它使用神经网络来建模复杂模式。
#句子4：近年来，深度学习在图像识别和自然语言处理上取得了巨大成功。
#得到 这种结构
{
  "text": "深度学习是机器学习的一个子领域。",
  "metadata": {
      "original_text": "深度学习是机器学习的一个子领域。",
      "window": "机器学习是一门研究如何让计算机从数据中学习的学科。 深度学习是机器学习的一个子领域。 它使用神经网络来建模复杂模式。"
  }
}
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

# 4.2 常规分块索引
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 5. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 6. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")
