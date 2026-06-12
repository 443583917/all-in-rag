import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# 了解多模态处理模型图片和文本比较


#模型的预训练权重文件 Visualized_base_en_v1.5.pth
# Visualized_BGE多模态嵌入模型将。
# 文本和图片映射到同一个向量空间中，使得文本和图片之间可以进行相似度计算
model = Visualized_BGE(
    # ① 加载文本编码器（使用预训练的 bge-base-en-v1.5 文本模型）
    model_name_bge="BAAI/bge-base-en-v1.5",
    # ② 加载视觉编码器的权重（.pth 文件）
    # 
    model_weight=r"D:\GitHub\all-in-rag\models\bge\Visualized_base_en_v1.5.pth")
#调用 model.eval() 表示进入推理模式（关闭 dropout 等训练特有的行为）
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image=r"D:\GitHub\all-in-rag\data\C3\imgs\datawhale01.png")
    multi_emb_1 = model.encode(image=r"D:\GitHub\all-in-rag\data\C3\imgs\datawhale01.png", text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image=r"D:\GitHub\all-in-rag\data\C3\imgs\datawhale02.png")
    multi_emb_2 = model.encode(image=r"D:\GitHub\all-in-rag\data\C3\imgs\datawhale02.png", text="datawhale开源组织的logo")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
