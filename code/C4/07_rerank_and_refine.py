import os
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

# 导入ColBERT重排器需要的模块
from langchain_core.documents import BaseDocumentCompressor
from langchain_classic.retrievers.document_compressors.base import  DocumentCompressorPipeline
from langchain_core.documents import Document
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
# 使用千文代替自定义的ColBERT重排器
from FlagEmbedding import FlagReranker
# Pipeline RAG 指的是一种标准化的“检索增强生成”工作流，
# 把数据处理和问答分成两个阶段：离线索引（Pipeline）+ 在线检索生成。
# 它是最常见的 RAG 实现方式，强调用流水线式步骤把文档转化为向量并存储，
# 再在用户提问时检索相关内容交给大模型生成答案
# 其他还有 自然 rag,agent rag ，Hybrid(混合) RAG 
# ===============================
# Embedding 和 Reranker 的本质区别
# Embedding 检索是「一句话 → 一个向量」然后和已有的向量进行 COSINE或者是IP算法进行比较获取
# 返回的topk不一定是真正的topk(最相关数据排名)
# 而Reranker Reranker负责排序 返回真正符合要求的前几条 使用Reranker的大模型处理
# 在大模型Reranker后可以根据业务偏好需要再重新打分
class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = "bert-base-uncased"

        # 加载模型和分词器
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """ColBERT文本编码"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """ColBERT相似度计算（MaxSim操作）"""
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 应用文档mask
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 应用查询mask
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最终分数
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序"""
        if len(documents) == 0:
            return documents

        # 编码查询
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 编码文档
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 计算ColBERT相似度
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs


class Qwen3Reranker(BaseDocumentCompressor):
    """Qwen3-Reranker重排器"""

    reranker: FlagReranker

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        object.__setattr__(
            self,
            "reranker",
            FlagReranker(
                "Qwen/Qwen3-Reranker-0.6B",
                use_fp16=True
            )
        )

        print("Qwen3-Reranker加载完成")

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks=None,
    ) -> Sequence[Document]:

        if len(documents) == 0:
            return documents

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.reranker.compute_score(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # 取Top5
        reranked_docs = [
            doc
            for doc, _ in scored_docs[:5]
        ]

        return reranked_docs


# 初始化配置
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

llm = ChatOpenAI(
    temperature=0.1,# 创造性
    max_tokens=1000,# 最大输出 越小越快
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)

# 1. 加载和处理文档
loader = TextLoader(r"D:\GitHub\all-in-rag\data\C4\txt\ai.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2. 创建向量存储和基础检索器
# vectorstore检索器 基于向量相似度检索，把 query 转 embedding，在向量数据库里找最相似的文档。
# 适合语义搜索
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. 设置ColBERT重排序器
# 替换直接使用
#reranker = Qwen3Reranker()
reranker = ColBERTReranker()

# 标准的压缩器写法
# 4. 设置LLM压缩器 用大模型提取关键信息
# 为什么要设计压缩器
# 目的：提取关键信息，减少冗余。
# 压缩的对象：检索到的文档内容。
# 作用：在检索结果进入 LLM 前做“二次清洗”，保证上下文简洁且相关。
compressor = LLMChainExtractor.from_llm(llm)

# 5. 使用DocumentCompressorPipeline组装压缩管道
# 流程: ColBERT重排 -> LLM压缩
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 6. 创建最终的压缩检索器
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever 
)

# 7. 执行查询并展示结果
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询: {query}\n")

# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")
# base_retriever类型是(VectorStoreRetriever接口类型)提供了方法从向量库检索数据
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")

# 7.2 使用管道压缩器的最终结果
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
# final_retriever是ContextualCompressionRetriever(继承了base_retriever接口多了层重排压缩逻辑)
final_results = final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")
