"""LLM layer. Default is MOCK_LLM (deterministic, keyless, offline).

* MOCK_LLM=1: `generate_grounded` is an extractive generator - it can only
  return sentences that already exist in the retrieved context, ranked by
  embedding similarity to the question. It never invents text.
* MOCK_LLM=0: the same prompt is sent to a real model (OpenAI Chat Completions API
  via httpx) if OPENAI_API_KEY is set. Optional; nothing graded needs it.
"""
from __future__ import annotations

import json
import os

import config
from rag.chunking import split_sentences
from rag.embeddings import cosine, embed, embed_one

GROUNDED_PROMPT = """You are Practo's patient-support assistant.
Answer the question using ONLY the context below. If the context does not
contain the answer, reply exactly: "I don't know based on the clinic policy knowledge base."
Do not use outside knowledge. Keep the answer to at most 3 sentences.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return GROUNDED_PROMPT.format(context=ctx, question=question)


def _real_llm(prompt: str, max_tokens: int = 300) -> str:  # pragma: no cover - optional path
    import httpx

    key = os.environ["OPENAI_API_KEY"]
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": config.LLM_MODEL, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def mock_extractive_answer(question: str, contexts: list[str], max_sentences: int = 2) -> str:
    sentences: list[str] = []
    for c in contexts:
        for s in split_sentences(c):
            if s not in sentences:
                sentences.append(s)
    if not sentences:
        return ""
    q = embed_one(question)
    vecs = embed(sentences)
    ranked = sorted(zip(sentences, vecs), key=lambda sv: cosine(q, sv[1]), reverse=True)
    best = ranked[0][1]
    chosen = [ranked[0][0]]
    for s, v in ranked[1:]:
        if len(chosen) >= max_sentences:
            break
        if cosine(q, v) >= cosine(q, best) - 0.10:  # only near-equally relevant sentences
            chosen.append(s)
    return " ".join(chosen)


def generate_grounded(question: str, contexts: list[str]) -> tuple[str, str]:
    """Returns (answer, prompt). Mock mode never sees anything but `contexts`."""
    prompt = build_prompt(question, contexts)
    if config.MOCK_LLM:
        return mock_extractive_answer(question, contexts), prompt
    return _real_llm(prompt), prompt


def judge(prompt: str) -> dict:  # used only when MOCK_LLM=0
    text = _real_llm(prompt, max_tokens=200)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])
