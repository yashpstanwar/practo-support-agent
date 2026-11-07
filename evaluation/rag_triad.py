"""Task 13 - RAG triad (context relevance, groundedness, answer relevance) on
15 queries with an LLM-as-judge prompt, run under MOCK_LLM.

Under MOCK_LLM the judge prompt below is rendered for every metric (and saved
to transcripts) but scored by a deterministic rubric implementation that
follows the same instructions using embedding similarity:

  context_relevance = (# of the top-3 retrieved chunks with cos(query, chunk)
                       >= IDK_THRESHOLD) / 3
  groundedness      = (# answer sentences supported by the context: verbatim or
                       cos(sentence, best context sentence) >= 0.80) / (# sentences)
                       An abstention makes no claims -> 1.0.
  answer_relevance  = cos(query, answer) clipped to [0, 1] for answered
                       queries; for an abstention: 1.0 if the query is labelled
                       out-of-scope (abstaining is the correct answer), else 0.0.

With MOCK_LLM=0 the same prompts are sent to the real judge model instead.

Run:  python -m evaluation.rag_triad
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("AGENT_VERBOSE", "0")  # keep the eval transcript readable

import config
import llm
from agent import memory as mem
from agent.graph import run_agent
from guardrails import check_groundedness
from rag.embeddings import cosine, embed_one
from rag.generate import IDK_ANSWER
from rag.store import retrieve

JUDGE_PROMPT = """You are an impartial evaluator of a retrieval-augmented support assistant.
Metric: {metric}
Rubric: {rubric}
Question: {question}
Retrieved context:
{context}
Answer: {answer}
Return JSON only: {{"score": <float between 0 and 1>, "reason": "<one sentence>"}}"""

RUBRICS = {
    "context_relevance": "Fraction of retrieved passages that are relevant to the question.",
    "groundedness": "Fraction of the answer's claims that are directly supported by the retrieved context. "
                    "An explicit 'I don't know' makes no claims and scores 1.",
    "answer_relevance": "How directly the answer addresses the question. For an out-of-scope question, "
                        "correctly declining to answer scores 1.",
}

TEST_SET = [
    {"q": "How many days in advance can I book an appointment?", "topic": "appointment-booking policy", "oos": False},
    {"q": "Will I lose money if I cancel two hours before my slot?", "topic": "cancellation/rescheduling window", "oos": False},
    {"q": "What is the consultation fee range for a pediatrician?", "topic": "consultation-fee structure", "oos": False},
    {"q": "How do I file an insurance claim for my lab tests?", "topic": "insurance-claim process", "oos": False},
    {"q": "Can I get my regular medicine refilled without seeing the doctor again?", "topic": "prescription-refill policy", "oos": False},
    {"q": "How long do culture test reports take to come back?", "topic": "lab-test turnaround times", "oos": False},
    {"q": "Is a video consultation allowed for chest pain?", "topic": "telemedicine eligibility", "oos": False},
    {"q": "What should I do if someone has signs of a stroke?", "topic": "emergency-visit protocol", "oos": False},
    {"q": "Can support agents see my diagnosis or clinical notes?", "topic": "patient-data privacy policy", "oos": False},
    {"q": "Is a follow-up within a week of my consultation free?", "topic": "follow-up-visit discount policy", "oos": False},
    {"q": "How do I get a second opinion from another specialist?", "topic": "second-opinion process", "oos": False},
    {"q": "Who is eligible for a doctor's home visit?", "topic": "home-visit eligibility", "oos": False},
    {"q": "Who won the IPL in 2023?", "topic": "OUT-OF-SCOPE (sports)", "oos": True},
    {"q": "What dose of paracetamol should I take for a fever?", "topic": "OUT-OF-SCOPE (medical advice)", "oos": True},
    {"q": "refund?", "topic": "EDGE CASE (one-word, ambiguous)", "oos": False},
]


def mock_judge(metric: str, question: str, contexts: list[str], answer: str, oos: bool,
               sims: list[float], top1: float) -> dict:
    abstained = answer.strip() in (IDK_ANSWER,) or answer.startswith("I can't answer that reliably")
    if metric == "context_relevance":
        rel = sum(s >= config.IDK_THRESHOLD for s in sims)
        return {"score": round(rel / max(len(sims), 1), 3),
                "reason": f"{rel}/{len(sims)} chunks have cos >= {config.IDK_THRESHOLD}"}
    if metric == "groundedness":
        if abstained:
            return {"score": 1.0, "reason": "abstention - no claims to verify"}
        g = check_groundedness(answer, contexts, top1)
        sup = sum(x["supported"] for x in g["support"])
        return {"score": round(sup / max(len(g["support"]), 1), 3),
                "reason": f"{sup}/{len(g['support'])} answer sentences supported"}
    if metric == "answer_relevance":
        if abstained:
            return {"score": 1.0 if oos else 0.0,
                    "reason": "correct abstention on out-of-scope query" if oos else "abstained on an in-scope query"}
        s = max(0.0, min(1.0, cosine(embed_one(question), embed_one(answer))))
        return {"score": round(s, 3), "reason": f"cos(question, answer) = {s:.3f}"}
    raise ValueError(metric)


def main() -> None:
    print(f"MOCK_LLM={config.MOCK_LLM}  collection={config.AGENT_COLLECTION}  IDK_THRESHOLD={config.IDK_THRESHOLD}")
    print(f"{len(TEST_SET)} queries; topics covered: {sum(not t['oos'] for t in TEST_SET[:12])} required KB topics "
          f"+ {sum(t['oos'] for t in TEST_SET)} out-of-scope + 1 edge case\n")
    rows = []
    sample_prompt_printed = False
    for i, t in enumerate(TEST_SET, 1):
        mem.reset("triad-eval")
        resp = run_agent(t["q"], "triad-eval")
        hits = retrieve(t["q"], k=config.TOP_K)
        contexts = [h.text for h in hits]
        sims = [h.score for h in hits]
        scores = {}
        for metric, rubric in RUBRICS.items():
            prompt = JUDGE_PROMPT.format(metric=metric, rubric=rubric, question=t["q"],
                                         context="\n".join(f"- {c}" for c in contexts), answer=resp["answer"])
            if not sample_prompt_printed:
                print("Sample judge prompt (query 1, context_relevance):\n" + prompt + "\n")
                sample_prompt_printed = True
            if config.MOCK_LLM:
                verdict = mock_judge(metric, t["q"], contexts, resp["answer"], t["oos"], sims, hits[0].score)
            else:  # pragma: no cover - optional real judge
                verdict = llm.judge(prompt)
            scores[metric] = verdict
        rows.append((i, t, resp, scores))
        print(f"Q{i:02d} [{t['topic']}] {t['q']}")
        print(f"     answer: {resp['answer'][:150]}")
        for m, v in scores.items():
            print(f"     {m:<18} {v['score']:.3f}  ({v['reason']})")
        print()

    print("=" * 96)
    print(f"{'#':<4}{'topic':<36}{'ctx_rel':>9}{'grounded':>10}{'ans_rel':>9}  route")
    print("-" * 96)
    avg = {m: 0.0 for m in RUBRICS}
    for i, t, resp, s in rows:
        for m in RUBRICS:
            avg[m] += s[m]["score"]
        print(f"Q{i:02d} {t['topic'][:34]:<36}{s['context_relevance']['score']:>9.3f}"
              f"{s['groundedness']['score']:>10.3f}{s['answer_relevance']['score']:>9.3f}  {resp['route']}")
    print("-" * 96)
    n = len(rows)
    print(f"{'AVG':<40}{avg['context_relevance'] / n:>9.3f}{avg['groundedness'] / n:>10.3f}"
          f"{avg['answer_relevance'] / n:>9.3f}")
    out = {"averages": {m: round(v / n, 4) for m, v in avg.items()},
           "per_query": [{"q": t["q"], "topic": t["topic"], "answer": r["answer"],
                          **{m: s[m]["score"] for m in RUBRICS}} for _, t, r, s in rows]}
    (config.ROOT / "evaluation" / "rag_triad_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved evaluation/rag_triad_results.json")


if __name__ == "__main__":
    main()
