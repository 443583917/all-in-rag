from langchain_text_splitters import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker 语义驱动的文本切分器
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)
#percentile → 按百分位数阈值，比如差异超过前 90% 的情况就切。
#standard_deviation → 按标准差阈值，差异超过平均值 ± nσ 就切。
#interquartile → 按四分位间距（IQR），差异超过 Q3+1.5IQR 就切。
#gradient → 按语义差异的梯度变化，找到变化最剧烈的点来切
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')

#切词器总结
#CharacterTextSplitter	固定字符数	简单快速	容易截断语义
#RecursiveCharacterTextSplitter	优先按段落/句子/空格递归切	保持语义完整	仍然基于字符边界
#SemanticChunker	Embedding 语义差异	切分更智能，语义边界更自然	速度慢，需要算 embedding   