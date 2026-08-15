"""
graph/nodes.py

Each function here is one "node" in the LangGraph pipeline. Every node takes
the current GraphState, does one job, and returns the fields it updated.
All LLM calls go through Groq (openai/gpt-oss-120b) - free and fast.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

from graph.state import GraphState
from graph.retriever import hybrid_retrieve

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 1


def _call_llm(system_prompt: str, user_prompt: str, _attempt: int = 0) -> str:
    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Handle transient per-minute rate limits with a short backoff (max 2 retries).
        # A daily quota (TPD) error won't be fixed by waiting a few seconds, so we
        # let that raise immediately and the eval script logs it and moves on.
        if "rate_limit" in str(e).lower() and _attempt < 2 and "per day" not in str(e).lower():
            wait_seconds = 3 * (_attempt + 1)
            print(f"    Rate limited, waiting {wait_seconds}s and retrying...")
            time.sleep(wait_seconds)
            return _call_llm(system_prompt, user_prompt, _attempt + 1)
        raise


# ---------------------------------------------------------------------------
# 1. ROUTER - classifies the query
# ---------------------------------------------------------------------------
def route_query(state: GraphState) -> dict:
    system = (
        "You classify user questions into exactly one category. "
        "Reply with ONLY one word, no punctuation, no explanation:\n"
        "- casual: greetings, small talk, no document lookup needed\n"
        "- simple: a single factual question answerable from one part of a document\n"
        "- complex: a question requiring comparing/combining multiple parts of documents\n"
        "- current_info: asks about something time-sensitive or outside any document's scope"
    )
    result = _call_llm(system, state["question"]).lower().strip()
    if result not in {"casual", "simple", "complex", "current_info"}:
        result = "simple"  # safe fallback
    return {"route": result, "retry_count": 0}


# ---------------------------------------------------------------------------
# 2. RETRIEVER - hybrid search + rerank
# ---------------------------------------------------------------------------
def retrieve(state: GraphState) -> dict:
    query = state.get("query_for_retrieval") or state["question"]
    chunks = hybrid_retrieve(query, top_k=5)
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# 3. GRADER - judges relevance, triggers query rewrite if needed
# ---------------------------------------------------------------------------
def grade_chunks(state: GraphState) -> dict:
    chunks_text = "\n\n".join(c["text"][:600] for c in state.get("retrieved_chunks", []))
    system = (
        "You judge whether retrieved document excerpts contain enough information "
        "to reasonably answer the user's question, even partially. Be lenient: if "
        "the excerpts are on-topic and contain relevant facts, answer 'yes'. Only "
        "answer 'no' if the excerpts are clearly about a different topic entirely. "
        "Reply with ONLY 'yes' or 'no'."
    )
    user = f"Question: {state['question']}\n\nRetrieved excerpts:\n{chunks_text}"
    result = _call_llm(system, user).lower().strip()
    relevant = result.startswith("y")
    return {"grader_relevant": relevant}


def rewrite_query(state: GraphState) -> dict:
    system = (
        "The retrieved documents were not relevant enough. Rewrite the user's "
        "question to be clearer and more specific, to improve search results. "
        "Reply with ONLY the rewritten question, nothing else."
    )
    rewritten = _call_llm(system, state["question"])
    return {
        "query_for_retrieval": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 4. MULTI-HOP DECOMPOSITION - for complex queries
# ---------------------------------------------------------------------------
def decompose_query(state: GraphState) -> dict:
    system = (
        "Break the user's complex question into 2-4 simpler sub-questions that, "
        "together, would let someone answer the original question. "
        'Reply with ONLY a JSON list of strings, e.g. ["sub-question 1", "sub-question 2"]'
    )
    raw = _call_llm(system, state["question"])
    try:
        sub_questions = json.loads(raw)
        assert isinstance(sub_questions, list)
    except Exception:
        sub_questions = [state["question"]]  # fallback: treat as single question
    return {"sub_questions": sub_questions}


def answer_sub_questions(state: GraphState) -> dict:
    sub_answers = []
    for sub_q in state.get("sub_questions", []):
        chunks = hybrid_retrieve(sub_q, top_k=3)
        context = "\n\n".join(c["text"] for c in chunks)
        system = "Answer the question using ONLY the provided context. Be concise."
        user = f"Context:\n{context}\n\nQuestion: {sub_q}"
        answer = _call_llm(system, user)
        sub_answers.append({"question": sub_q, "answer": answer, "chunks": chunks})
    return {"sub_answers": sub_answers}


# ---------------------------------------------------------------------------
# 5. GENERATOR - final answer with citations
# ---------------------------------------------------------------------------
def generate_answer(state: GraphState) -> dict:
    if state.get("sub_answers"):
        # complex path: synthesize from sub-answers
        context = "\n\n".join(
            f"Q: {sa['question']}\nA: {sa['answer']}" for sa in state["sub_answers"]
        )
        all_chunks = [c for sa in state["sub_answers"] for c in sa["chunks"]]
    else:
        # simple path: use retrieved chunks directly
        context = "\n\n".join(c["text"] for c in state.get("retrieved_chunks", []))
        all_chunks = state.get("retrieved_chunks", [])

    sources = sorted(set(c["source"] for c in all_chunks))

    system = (
        "Answer using ONLY the given context. Be clear and direct. "
        "Do not mention sources or where info came from."
    )
    user = f"Context:\n{context}\n\nQuestion: {state['question']}"
    answer = _call_llm(system, user)

    return {"answer": answer, "citations": sources}


# ---------------------------------------------------------------------------
# 6. GROUNDEDNESS CHECKER - catches hallucinations
# ---------------------------------------------------------------------------
def check_groundedness(state: GraphState) -> dict:
    if state.get("sub_answers"):
        context = "\n\n".join(
            f"Q: {sa['question']}\nA: {sa['answer']}" for sa in state["sub_answers"]
        )
    else:
        context = "\n\n".join(c["text"] for c in state.get("retrieved_chunks", []))

    system = (
        "You check whether an answer is fully supported by the given context. "
        "Reply with ONLY 'yes' if fully supported, or 'no: <short reason>' if not."
    )
    user = f"Context:\n{context}\n\nAnswer to check:\n{state['answer']}"
    result = _call_llm(system, user).strip()

    passed = result.lower().startswith("yes")
    note = "" if passed else result

    return {"groundedness_pass": passed, "groundedness_note": note}


# ---------------------------------------------------------------------------
# 7. CASUAL RESPONSE - no retrieval needed
# ---------------------------------------------------------------------------
def casual_response(state: GraphState) -> dict:
    system = "You are a friendly assistant. Respond briefly and naturally."
    answer = _call_llm(system, state["question"])
    return {"answer": answer, "citations": [], "groundedness_pass": True}