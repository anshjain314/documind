"""
eval/run_eval.py

Runs a set of test questions through the full pipeline and reports:
- Groundedness pass rate (how often the answer was verified as supported by context)
- Route distribution (how many questions hit each branch)
- Retry rate (how often the grader had to trigger a query rewrite)

Edit EVAL_QUESTIONS below with 15-30 real questions based on YOUR PDFs
before running this for real numbers.

Run from the project root:
    python -m eval.run_eval
"""

from graph.build_graph import compiled_graph

# ---------------------------------------------------------------------------
# EDIT THIS: replace with real questions based on YOUR uploaded PDFs.
# Mix of simple, complex, casual, and "not in the documents" questions
# gives the most convincing, honest eval numbers.
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    # --- Simple factual ---
    "what is biogas made of?",
    "what is the ideal pH range for biogas generation?",
    "what is the Betz limit?",
    "what is tidal barrage?",
    "what is OTEC?",
    "how deep are hydrothermal resources typically located?",
    "what is the cut-in wind speed for most turbines?",

    # --- Complex / multi-hop ---
    "compare the working principles of Darrieus and Savonius wind turbines",
    "compare vapour dominated and liquid dominated geothermal systems",
    "compare open cycle and closed cycle OTEC systems",

    # --- Casual ---
    "hi, how's it going?",

    # --- Deliberately NOT in the documents ---
    "what is the capital of France?",
]


def run_eval():
    results = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")
        try:
            result = compiled_graph.invoke({"question": question})
            results.append({
                "question": question,
                "route": result.get("route"),
                "retry_count": result.get("retry_count", 0),
                "groundedness_pass": result.get("groundedness_pass"),
                "answer_preview": (result.get("answer") or "")[:100],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": question,
                "route": "ERROR",
                "retry_count": 0,
                "groundedness_pass": False,
                "answer_preview": str(e)[:100],
            })

    # --- Report ---
    total = len(results)
    grounded_pass = sum(1 for r in results if r["groundedness_pass"])
    retried = sum(1 for r in results if r["retry_count"] > 0)

    route_counts = {}
    for r in results:
        route_counts[r["route"]] = route_counts.get(r["route"], 0) + 1

    print("\n" + "=" * 50)
    print("EVAL REPORT")
    print("=" * 50)
    print(f"Total questions tested:     {total}")
    print(f"Groundedness pass rate:     {grounded_pass}/{total} ({100*grounded_pass/total:.1f}%)")
    print(f"Questions that triggered a retry: {retried}/{total} ({100*retried/total:.1f}%)")
    print(f"Route distribution:         {route_counts}")
    print("=" * 50)

    print("\nPer-question detail:")
    for r in results:
        status = "PASS" if r["groundedness_pass"] else "FAIL"
        print(f"[{status}] ({r['route']}, retries={r['retry_count']}) {r['question']}")
        print(f"       -> {r['answer_preview']}...")

    return results


if __name__ == "__main__":
    run_eval()