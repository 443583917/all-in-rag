import os

# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

markdown_path = r"D:\GitHub\all-in-rag\data\C1\markdown\easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredLoader(markdown_path)
docs = loader.load()

# 文本分块
# RecursiveCharacterTextSplitter( 可以设置的参数
# chunk_size=500, 每个 chunk 的最大长度 不填默认1000
# chunk_overlap=50  相邻 chunk 的重叠部分长度。
# separators=["\n\n", "\n", "。", " "] )# 优先按段落/句子切
# embedding 模型通常只能处理有限长度（512～1024 token）
# chunk 越小，向量库越大；chunk 越大，检索噪音越多。
text_splitter = RecursiveCharacterTextSplitter()
# 返回的 chunks 就是一个个小文本片段，
# 每个片段都带有原始文档的 metadata（比如来源、页码）。
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
# BAAI/bge-small-zh-v1.5 是北京智源人工智能研究院开发的
# 专门做文本向量的中文模型(不同模型生成的向量不同不能混用)
# bge-small-zh-v1.5 维度是 384 维度越低 检索越快 维度高检索慢适用高质量检索
# 目的是把文本映射到低维稠密向量
# embedding(嵌入)把自然语言变成一个“能被计算机理解的数字向量”
# 相似的句子 → 向量距离更近,不相似的句子 → 距离更远 这就是语义搜索的基础.
embeddings = HuggingFaceEmbeddings( # HuggingFaceEmbeddings 类的实例 两个方法
                                    #embeddings.embed_documents：把一组文档转成向量
                                    #embeddings.embed_query：把用户查询转成向量
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

#RAG（检索增强生成） 
# 流程：
# 文档 → 切 chunk  chunk = 文本片段（通常 200～500 字）
# chunk → embedding
# 用户问题 → embedding
# 两个向量做相似度匹配
# 找到最相关的文档
# 送给大模型回答


# 构建向量存储
#创建的 vectorstore 是一个 内存向量数据库对象。可以存放chunks转换后的向量数据
vectorstore = InMemoryVectorStore(embeddings)
#add_documents(chunks)  将chunks中的每个 chunk 转成 embedding 向量，并存入 vectorstore
vectorstore.add_documents(chunks)
# 用完即丢

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:/
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    temperature=0.7,# 创造性
    max_tokens=4096,# 最大输出 越小越快
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)


# 用户查询
question = "强化学习和监督学习的区别"

# 在向量存储中查询相关文档
# k 值 表示要返回的 最相似文档片段数量
retrieved_docs = vectorstore.similarity_search(question, k=3)
# vectorstore的方法作用和用法介绍
# vectorstore.similarity_search("用户问题", k=3) 输出：最相似的 Document 列表
# vectorstore.similarity_search_with_score("问题", k=3) 输出 (Document, 相似度分数) 的列表
# vectorstore.max_marginal_relevance_search("问题", k=3, fetch_k=10) 先找10个最后挑3个最好的
# 输出：一组去重后的相关文档，避免重复信息
# vectorstore.add_documents(chunks) 把文档转成向量并存入库
# vectorstore.add_texts(["文本1", "文本2"], metadatas=[{"id":1},{"id":2}]) 
# 把文本和对应额外信息一起存入，能够在使用时更好的知道数据来源
# doc 是从 retrieved_docs 列表里迭代出来的对象
# 迭代器写法 for doc in retrieved_docs
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
