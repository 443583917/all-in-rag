import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image
# 大企业级别向量库
# Visualized-BGE多模态嵌入模型（文本 + 图像）,HuggingFaceEmbeddings 纯文本模型
# 实现了一个完整的 多模态图文检索流程，结合了 Visualized-BGE 模型 和 Milvus 向量数据库
# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = r"D:\GitHub\all-in-rag\models\bge\Visualized_base_en_v1.5.pth"
DATA_DIR = r"D:\GitHub\all-in-rag\data\C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"

# 2. 定义工具 (编码器和可视化函数)
# self 指向实例本身可以直接调用方法 Encoder.encode_query。有点像go方法类型
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):
        # self.model model字段是声明了就直接创建就有Visualized_BGE属性了。
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval() #进入“推理模式”
        # 模型有推理模型和训练模式,使用时选择推理模式
    # self方法中的__init__需要先Encoder(MODEL_NAME, MODEL_PATH)初始化后才能调用后面定义发方法
    # 把图像 + 文本联合转成查询向量。
    def encode_query(self, image_path: str, text: str) -> list[float]:
        # with torch.no_grad()表示推理时不计算梯度
        # 在推理模式下进行预测，只生成向量，不做任何学习。
        with torch.no_grad():
            #query_emb中间变量为了好读代码
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]
    # 把图像转成向量。
    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]
# 把查询图像和检索到的图像拼接成一个全景图，用于可视化展示
def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])

# 3. 初始化客户端
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
# Collection 等同与数据库中的表,MilvusZ中的表不能重名所以要先判断以及删除
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")
# 获取存放图片的路径list
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
# 用len来计算维度
dim = len(encoder.encode_image(image_list[0]))

fields = [
    #Milvus Collection 的字段结构 (schema)
    #在 Milvus 里，Collection 就像数据库的表，而 FieldSchema 就是表里的“列定义”。
    #让llm 自动生成的 ID、存储向量的列、存储图片路径的列。
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 创建集合 Schema 相当表结构
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# 5. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = [] # 空列表 批量插入数据快
# tqdm 进度条迭代器。更直观看到图片的向量转换
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    # 选择插入的collection_name(集合名称)和数据data
    # milvus_client执行完插入会得到操作后的结构字段，保存了本次插入的操作结果
    # insert_count → 实际成功插入的数据条数。
    # ids → 插入数据的主键 ID 列表
    # status → 插入操作是否成功。
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# 6. 创建索引 index_params索引参数对象
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
# 只有数据插入完成后才能为需要的字段创建索引。一般都是vector(向量)字段创建
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",#索引类型 HNSW领近搜索
    metric_type="COSINE",# 指定相似度度量方式为 余弦相似度。
    params={"M": 16, "efConstruction": 256}#HNSW 的参数
    #M：每个节点的最大邻居数，控制图的稠密程度。
    # efConstruction：建索引时的搜索范围，值越大索引质量越好，但建索引更慢。
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
#后续再插入新数据 collection.insert(new_data) 会自动维护索引
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 7. 执行多模态检索 用图像和文本作为查询向量
# 单模态检索：只用一种模态（比如只用文本，或者只用图像）去查找相似内容。
# 多模态检索：同时结合多种模态（图像 + 文本），生成一个联合向量，在数据库里查找最相似的结果。
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# 一个标准的milvus_client查询
# 指定查询的集合名称，查询的条件向量
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],#默认返回的只有相识度评分和对应的id。需要返回额外字段在此说明
    limit=5,#只返回前 5 个最相似的结果 不限制会返回非常多
    #metric_type 索引和检索必须一致，不能随意换
    #ef越小越快
    # 检索时的参数，可以比 efConstruction 小，也可以比它大，影响查询精度和速度
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0] 
#search_results的结构是二维数组,支持查询多个条件向量查询。因为这边只有一个条件所以只取[0] 获取到对于list
#取回图片列表
retrieved_images = []
print("检索结果:")
# enumerate遍历函数
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# 8. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    #visualize_results的内置方法，能把要搜索的图片和搜索出的图片够成一张图进行对比方便观看
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")
