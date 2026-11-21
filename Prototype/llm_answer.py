"""Write a short answer from the retrieved manual pages, and check every quote.

The language model may only use the numbered sources. For each point it must give the
source number and a word-for-word quote. We then check that the quote really appears in
that source. Points whose quote can not be found are removed.
"""
import json
import re
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Tried in order. Groq changes its model list, so the first working one is used.
MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]

SYSTEM = """You help plant operators troubleshoot instruments and valves.
Use ONLY the numbered sources you are given. Do not use any other knowledge.
Reply with JSON only, in this exact form:
{"points": [{"text": "one short sentence", "source": 1, "quote": "words copied exactly from source 1"}]}
Rules:
- Give at most 4 points.
- Every point needs the source number and a quote of 6 to 25 words, copied word for word from that source.
- If the sources do not answer the question, reply {"points": []}.
- Never invent page numbers, values or steps."""


def build_messages(situation, sources):
    blocks = [f"[{i}] {s['title']}, {s.get('section') or 'no section'}, page {s['page']}\n{s['text']}"
              for i, s in enumerate(sources, start=1)]
    user = f"{situation}\n\nSources:\n\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask_groq(api_key, messages, preferred=None, retries=3):
    """Call Groq. Returns (reply_text, model_used).
    A rate limit (HTTP 429) is waited out and retried; a missing model (404) or any other error moves on to the next model."""
    models = ([preferred] if preferred else []) + [m for m in MODELS if m != preferred]
    errors = []
    for model in models:
        for attempt in range(retries):
            try:
                r = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages, "temperature": 0, "max_tokens": 2000},
                    timeout=60,
                )
            except requests.RequestException as e:
                errors.append(f"{model}: {e}")
                break
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], model
            if r.status_code == 429 and attempt < retries - 1:
                try:
                    wait = float(r.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0
                time.sleep(min(wait or 15 * (attempt + 1), 60))
                continue
            errors.append(f"{model}: HTTP {r.status_code} {r.text[:120]}")
            break
    raise RuntimeError(" | ".join(errors) or "no model tried")


def squash(text):
    """Lower-case letters and digits only, so spacing and hyphenation do not matter."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_points(reply):
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        points = json.loads(reply[start:end + 1]).get("points", [])
    except json.JSONDecodeError:
        return []
    return points if isinstance(points, list) else []


def verify(points, sources):
    """Keep only points whose quote appears word for word in the cited source."""
    kept, dropped = [], 0
    for p in points:
        try:
            n = int(p["source"])
            quote, text = str(p["quote"]), str(p["text"])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        ok = 1 <= n <= len(sources) and len(squash(quote)) >= 25 and squash(quote) in squash(sources[n - 1]["text"])
        if ok:
            kept.append({"text": text, "source": n, "quote": quote})
        else:
            dropped += 1
    return kept, dropped


def answer(api_key, situation, sources, preferred=None):
    reply, model = ask_groq(api_key, build_messages(situation, sources), preferred)
    kept, dropped = verify(parse_points(reply), sources)
    return {"points": kept, "dropped": dropped, "model": model}
