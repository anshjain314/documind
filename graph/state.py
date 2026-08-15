"""
graph/state.py

The shared state object that flows through every node in the LangGraph pipeline.
Think of this as a clipboard that gets passed from node to node, with each
node reading what it needs and adding/updating fields.
"""

from typing import TypedDict, Optional


class GraphState(TypedDict, total=False):
    question: str                 # the user's original question
    route: str                    # "casual" | "simple" | "complex" | "current_info"
    sub_questions: list[str]      # for complex/multi-hop queries
    query_for_retrieval: str      # possibly rewritten query used for retrieval
    retrieved_chunks: list[dict]  # chunks after hybrid search + rerank
    grader_relevant: bool         # did the grader judge chunks as relevant?
    retry_count: int              # how many retrieval retries have happened
    sub_answers: list[dict]       # {"question": ..., "answer": ..., "chunks": [...]}
    answer: str                   # final generated answer
    citations: list[str]          # source names cited in the answer
    groundedness_pass: bool       # did the answer pass the hallucination check?
    groundedness_note: str        # explanation if it failed
