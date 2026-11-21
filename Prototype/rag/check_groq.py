"""Which Groq models can this key use right now? Prints one line per model, never the key.

    $env:GROQ_API_KEY = "your_key_here"
    python rag/check_groq.py
"""
import os
import sys

import requests

sys.path.insert(0, ".")
from llm_answer import GROQ_URL, MODELS

key = os.environ.get("GROQ_API_KEY") or sys.exit("Set GROQ_API_KEY first")
headers = {"Authorization": f"Bearer {key}"}
r = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=30)
print("model list:", r.status_code)
if r.status_code == 200:
    available = sorted(m["id"] for m in r.json()["data"])
    print("available:", ", ".join(available))
for model in MODELS:
    r = requests.post(GROQ_URL, headers=headers, timeout=60,
                      json={"model": model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 20})
    print(f"{model}: HTTP {r.status_code}", "" if r.status_code == 200 else r.text[:200])
