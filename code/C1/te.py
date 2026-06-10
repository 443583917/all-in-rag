import requests

url = "http://127.0.0.1:8080/v1/embeddings"
payload = {"model": "Qwen/Qwen3-Embedding-0.6B", "input": "你好"}
resp = requests.post(url, json=payload)
print(resp.json())
