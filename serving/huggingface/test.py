import requests


url = "http://127.0.0.1:8079/chat/completions/"
data = {
    "messages": [
        {"role": "system", "content": "Bạn là một con bot hiểu tiếng người"},
        {"role": "user", "content": "Mày là thằng nào?"}
    ],
    "stream": True
}

with requests.post(url, stream=True, json=data) as r:
    for chunk in r.iter_content(1024):
        print(chunk.decode("utf-8"), sep="")

# r = requests.post(url, json=data)
# print(r.text)
