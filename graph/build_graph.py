"""
graph/build_graph.py

Wires all the nodes together into the actual LangGraph state machine.
This is the file that defines HOW the pipeline flows: which node runs
next depends on the route classification, the grader's verdict, and
the retry count.
"""

import os
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from graph.state import GraphState
from graph import nodes

load_dotenv()

# Enable LangSmith tracing automatically via env vars (LANGCHAIN_TRACING_V2=true)
# No extra code needed here - LangGraph picks it up from the environment.


def route_after_classification(state: GraphState) -> str:
    """Decide which path to take based on the router's classification."""
    route = state.get("route", "simple")
    if route == "casual":
        return "casual_response"
    if route == "complex":
        return "decompose_query"
    # "simple" and "current_info" both go through standard retrieval for now
    return "retrieve"


def route_after_grading(state: GraphState) -> str:
    """Decide whether to proceed to generation or retry retrieval."""
    if state.get("grader_relevant"):
        return "generate_answer"
    if state.get("retry_count", 0) >= nodes.MAX_RETRIES:
        # give up retrying, answer with what we have rather than looping forever
        return "generate_answer"
    return "rewrite_query"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_query", nodes.route_query)
    graph.add_node("casual_response", nodes.casual_response)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_chunks", nodes.grade_chunks)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("decompose_query", nodes.decompose_query)
    graph.add_node("answer_sub_questions", nodes.answer_sub_questions)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("check_groundedness", nodes.check_groundedness)

    graph.set_entry_point("route_query")

    # after classifying, branch to casual / retrieve / decompose
    graph.add_conditional_edges(
        "route_query",
        route_after_classification,
        {
            "casual_response": "casual_response",
            "retrieve": "retrieve",
            "decompose_query": "decompose_query",
        },
    )

    # simple/current_info path: retrieve -> grade -> (retry or generate)
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")  # loop back after rewriting

    # complex path: decompose -> answer each sub-question -> generate
    graph.add_edge("decompose_query", "answer_sub_questions")
    graph.add_edge("answer_sub_questions", "generate_answer")

    # after generating, always check groundedness, then end
    graph.add_edge("generate_answer", "check_groundedness")
    graph.add_edge("check_groundedness", END)

    # casual path skips straight to the end (no groundedness check needed)
    graph.add_edge("casual_response", END)

    return graph.compile()


# Build once, reuse across calls (e.g. from the Streamlit app)
compiled_graph = build_graph()


if __name__ == "__main__":
    # Quick manual test from the command line
    test_question = input("Ask DocuMind a question: ")
    result = compiled_graph.invoke({"question": test_question})
    print("\n--- ROUTE ---")
    print(result.get("route"))
    print("\n--- ANSWER ---")
    print(result.get("answer"))
    print("\n--- CITATIONS ---")
    print(result.get("citations"))
    print("\n--- GROUNDEDNESS ---")
    print(result.get("groundedness_pass"), result.get("groundedness_note"))
