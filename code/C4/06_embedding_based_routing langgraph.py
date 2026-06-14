import os
import numpy as np
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity

# 1. 定义路由描述
route_prompts = [
    "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。",
    "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"
]
route_names = ["川菜", "粤菜"]

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
route_prompt_embeddings = embeddings.embed_documents(route_prompts)

# 2. 初始化 LLM
llm = ChatOpenAI(
    temperature=0.1,
    max_tokens=1000,
    model="mimo-v2-flash",
    api_key="sk-ct6ct1y17ry3m9xh2rce3bbx68kbsqs19y326ym89hxw2k64",
    base_url="https://api.xiaomimimo.com/v1",
)

# 3. 定义节点函数
def route_node(state: dict):
    """路由节点：根据用户问题选择川菜或粤菜"""
    query = state["query"]
    query_embedding = embeddings.embed_query(query)
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    chosen_route_index = np.argmax(similarity_scores)
    state["route"] = route_names[chosen_route_index]
    print(f"路由决策: {state['route']}")
    return state

def sichuan_node(state: dict):
    """川菜节点：调用川菜链回答问题"""
    query = state["query"]
    prompt = PromptTemplate.from_template(
        "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。"
    )
    formatted = prompt.format(query=query)
    result = llm.invoke(formatted)
    state["answer"] = StrOutputParser().invoke(result)
    return state

def cantonese_node(state: dict):
    """粤菜节点：调用粤菜链回答问题"""
    query = state["query"]
    prompt = PromptTemplate.from_template(
        "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。"
    )
    formatted = prompt.format(query=query)
    result = llm.invoke(formatted)
    state["answer"] = StrOutputParser().invoke(result)
    return state

# 4. 构建 Graph
graph = StateGraph(dict)

graph.add_node("route", route_node)
graph.add_node("sichuan", sichuan_node)
graph.add_node("cantonese", cantonese_node)

graph.set_entry_point("route")

# 路由分支：根据 route 决策跳转
graph.add_conditional_edges(
    "route",
    lambda state: state["route"],
    {
        "川菜": "sichuan",
        "粤菜": "cantonese",
    }
)

graph.add_edge("sichuan", END)
graph.add_edge("cantonese", END)

app = graph.compile()

# 5. 演示查询
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

for q in demo_queries:
    result = app.invoke({"query": q})
    print(f"\n问题: {q}")
    print(f"路由: {result['route']}")
    print(f"回答: {result['answer']}")
