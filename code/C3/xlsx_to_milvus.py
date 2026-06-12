import hashlib
import pandas as pd
import time
import os
from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "movie_collection"
client = MilvusClient(uri=MILVUS_URI)

# 第一步：创建 Collection（表结构）
if not client.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=200),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="director", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="actors", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rating_count", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512)  # ✅ 注意新版要用 dimension
    ]
    schema = CollectionSchema(fields=fields, description="电影知识库")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print("第一步完成：Collection 创建成功")
# ✅ 要确认建表时vector维度和模型维度保持一致
# 第二步：初始化 Embedding 模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
print(embed_model.model_name)
print(len(embed_model.get_text_embedding("你好")))
# 第三步：读取 Excel 文件
excel_file = r"D:\GitHub\all-in-rag\data\C3\excel\movie.xlsx"
xls = pd.ExcelFile(excel_file)

# 第四步：批量插入准备
batch_data = []
batch_size = 64

for sheet_name in xls.sheet_names:
    print(f"\n处理 Sheet：{sheet_name}")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 第五步：数据清洗（评分人数列）
    if "评分人数" in df.columns:
        df["评分人数"] = (
            df["评分人数"].astype(str).str.replace("人评价", "").str.strip()
        )
        df["评分人数"] = (
            pd.to_numeric(df["评分人数"], errors="coerce").fillna(0).astype(int)
        )

    texts = []
    rows_to_insert = []

    # 第六步：遍历每一行，生成文本描述 + 主键 + hash
    for row_index, row in df.iterrows():
        text = (
            f"电影《{row['电影名称']}》，导演 {row['导演']}，主演 {row['主演']}，"
            f"年份 {row['年份']}，国家 {row['国家']}，分类 {row['分类']}，评分人数 {row['评分人数']}。"
        )
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        record_id = f"{row['电影名称']}_{int(row['年份'])}"

        # 第七步：第一次创建不能使用query查询是否有重复
        texts.append(text)
        rows_to_insert.append((row_index, row, text, content_hash, record_id))

    # 第八步：批量生成向量
    if len(texts) == 0:
        continue
    vectors = embed_model.get_text_embedding_batch(texts)
    # 第九步：组装数据并批量写入
    for vector, item in zip(vectors, rows_to_insert):
        row_index, row, text, content_hash, record_id = item
        batch_data.append({
            "id": record_id,
            "document_id": sheet_name,
            "chunk_index": int(row_index),
            "content_hash": content_hash,
            "title": row["电影名称"],
            "director": row["导演"],
            "actors": row["主演"],
            "year": int(row["年份"]),
            "country": row["国家"],
            "category": row["分类"],
            "rating_count": int(row["评分人数"]),
            "text": text,
            "vector": vector
        })
        if len(batch_data) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, data=batch_data)
            print(f"批量写入 {len(batch_data)} 条")
            batch_data.clear()

# 第十步：写入最后一批
if batch_data:
    client.upsert(collection_name=COLLECTION_NAME, data=batch_data)
    print(f"批量写入 {len(batch_data)} 条")

# 第十一步：创建索引（向量检索必须有索引）
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)
print("索引创建完成")
# 第十二步：加载 Collection 到内存
client.load_collection(collection_name=COLLECTION_NAME)
print("Collection 已加载")

# 第十三步：查看总记录数
stats = client.get_collection_stats(COLLECTION_NAME)
print("\n===================")
print("总记录数:", stats["row_count"])
print("===================")

# 怎么更新或者删除
# 在 Milvus 里，更新和删除数据主要依赖主键 pk 来操作。因为 Milvus 本身是“插入型”的数据库，
# 不支持直接覆盖更新，所以常见做法是 先删除旧数据，再插入新数据。
# 删除指定主键的数据
#collection.delete(expr='pk == "肖申克的救赎_1994"')
#collection.flush()
# 更新一般都是先复制再删除