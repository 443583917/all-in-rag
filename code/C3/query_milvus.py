from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import ChatOpenAI

# ==========================================
# Milvus
# ==========================================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "movie_collection"
client = MilvusClient(uri=MILVUS_URI)
# ==========================================
# Embedding Model
# ==========================================
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
# ==========================================
# LLM
# ==========================================
llm = ChatOpenAI(
    temperature=0.1,# 创造性
    max_tokens=1000,# 最大输出 越小越快
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)
# ==========================================
# Retriever
# ==========================================
def retrieve(
        query: str,
        top_k: int = 5
):
    query_vector = embed_model.get_text_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=[
            "text"
        ]
    )
    return results[0]
# ==========================================
# RAG
# ==========================================
def rag(
        question: str
):
    docs = retrieve(question)
    context = "\n".join(
        doc["entity"]["text"]
        for doc in docs
    )
    response = llm.invoke(
        f"""
根据下面提供的电影资料回答问题。
电影资料：
{context}
问题：
{question}
如果资料中没有答案，请明确说明不知道，不要编造。
""" )
    return response.content
# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    while True:
        question = input("\n请输入问题(q退出)：")
        if question.lower() == "q":
            break
        answer = rag(question)
        print("\n====================")
        print(answer)
        print("====================")
