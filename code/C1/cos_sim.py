import requests
import numpy as np

# 调用 Docker 容器的 Embedding API，获取文本的向量表示
def get_embedding(text: str):
    url = "http://127.0.0.1:8080/v1/embeddings"   # 容器暴露的 API 地址
    payload = {
        "model": "Qwen/Qwen3-Embedding-0.6B",     # 指定使用的模型
        "input": text                             # 输入文本
    }
    response = requests.post(url, json=payload)   # 向 API 发送 POST 请求
    data = response.json()                        # 解析返回的 JSON 数据
    # 返回 embedding 向量（转为 numpy 数组方便后续计算）
    return np.array(data["data"][0]["embedding"])

# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    # 公式：cos_sim = (vec1 · vec2) / (||vec1|| * ||vec2||)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 比较 query 与候选文本列表的相似度
def compare_similarity(query: str, candidates: list[str], top_n: int = None):
    query_vec = get_embedding(query)              # 获取 query 的向量
    results = []
    for i, text in enumerate(candidates, start=1):
        cand_vec = get_embedding(text)            # 获取候选文本的向量
        sim = cosine_similarity(query_vec, cand_vec)  # 计算相似度
        results.append((i, text, sim))            # 保存结果 (编号, 文本, 相似度)

    # 按相似度从高到低排序
    results.sort(key=lambda x: x[2], reverse=True)

    # 如果指定了 top_n，只返回前 top_n 个结果；否则返回全部
    if top_n:
        results = results[:top_n]
    return results

# ===== 测试 =====
query = "我真没想这样啊"   # 查询文本
candidates = [
    "我真想这样",
    "我不想这样",
    "这是迫不得已的情况",
    "没办法只能怎么做了也是不得不",
    "今天下雨我真有带雨伞，没有吃饭"
]

# 设置 top_n=5，返回所有候选的相似度
results = compare_similarity(query, candidates, top_n=5)

# 输出结果
for idx, text, sim in results:
    print(f"候选 {idx}: 相似度={sim:.4f}\n内容: {text}\n")
