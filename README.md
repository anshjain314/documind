# DocuMind — Agentic Multi-Document RAG Assistant

DocuMind is a document Q&A assistant that goes beyond basic RAG (Retrieval-Augmented Generation). Instead of a single fixed retrieve-then-answer pass, it routes each question through an adaptive pipeline that decides how much work a question actually needs, checks its own retrieval quality before answering, and verifies its own answers are actually supported by the source documents before returning them to the user.

Built as a demonstration of production-style agentic RAG patterns: adaptive routing, hybrid search, reranking, self-grading retrieval loops, multi-hop query decomposition, and groundedness verification.

## What it does (plain English)

You upload documents (in this case, three sets of engineering lecture notes on renewable energy). You ask a question in plain English. Instead of just grabbing the nearest-sounding text and answering immediately, the system:

1. **Decides if it even needs to search** — a greeting like "hi" skips document lookup entirely.
2. **Classifies the question's complexity** — simple factual questions get a direct search; questions that require comparing multiple concepts get broken into sub-questions first.
3. **Searches two ways at once** — semantically (meaning-based, via embeddings) and by keyword (exact-term matching), then combines and reranks the results — catching things a single search method would miss.
4. **Checks its own retrieval** — an LLM judges whether what was retrieved actually looks relevant. If not, it rewrites the query and searches again (up to a retry cap).
5. **Checks its own answer** — after generating a response, a separate check verifies the answer is actually supported by the retrieved text, rather than the model making something up.

## Architecture

```
                     ┌──────────────┐
   question ────────▶│    Router    │
                     └──────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     ┌─────────┐      ┌───────────┐    ┌──────────────┐
     │ Casual  │      │  Retrieve │    │  Decompose    │
     │response │      │(hybrid +  │    │ (multi-hop)   │
     └────┬────┘      │ rerank)   │    └──────┬───────┘
          │           └─────┬─────┘           │
          │                 ▼                 ▼
          │           ┌───────────┐    ┌──────────────┐
          │           │  Grader   │    │Answer each   │
          │           └─────┬─────┘    │sub-question  │
          │        no │     │ yes      └──────┬───────┘
          │           ▼     │                 │
          │      ┌─────────┐│                 │
          │      │ Rewrite ││                 │
          │      │ query   ││                 │
          │      └────┬────┘│                 │
          │           └─loop┘                 │
          │                 │                 │
          │                 ▼                 ▼
          │           ┌─────────────────────────┐
          │           │       Generator          │
          │           └────────────┬────────────┘
          │                        ▼
          │           ┌─────────────────────────┐
          │           │  Groundedness Checker    │
          │           └────────────┬────────────┘
          │                        │
          └────────────────────────┴──▶ final answer
```

**Nodes:**
- **Router** — classifies each question as `casual`, `simple`, `complex`, or `current_info`
- **Retriever** — hybrid search: semantic (Pinecone, via `llama-text-embed-v2` integrated embedding) + keyword (BM25), fused with reciprocal rank fusion, then reranked by Cohere `rerank-v3.5`
- **Grader** — judges retrieval relevance; triggers a query rewrite + retry (capped at 1) if the retrieved chunks don't look sufficient
- **Decomposer** — for complex questions, breaks the query into 2-4 sub-questions, each retrieved and answered independently
- **Generator** — synthesizes the final answer from retrieved context (or sub-answers, for complex queries)
- **Groundedness Checker** — verifies the final answer is actually supported by the retrieved context; flags it if not

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Stateful graph with conditional branching and retry loops |
| LLM inference | Groq (`llama-3.1-8b-instant`) | Free tier, fast inference |
| Embeddings | Pinecone integrated inference (`llama-text-embed-v2`) | No separate embedding API call needed; outperforms OpenAI's `text-embedding-3-large` on multiple benchmarks |
| Vector DB | Pinecone (free tier) | Managed, integrated embedding + search |
| Keyword search | BM25 (`rank-bm25`) | Catches exact-term matches semantic search can miss |
| Reranking | Cohere `rerank-v3.5` (free tier) | Fuses and reranks hybrid search candidates |
| Tracing | LangSmith (free tier) | Full visibility into every node execution and retry |
| UI | Streamlit | Fast to build, includes a live transparency panel showing the pipeline's decisions |

Entire stack runs on free tiers — no paid API usage required.

## Evaluation results

Ran against a 12-question test set spanning simple factual lookups, complex multi-hop comparisons, casual conversation, and a deliberately out-of-scope question (to test the groundedness check catches ungrounded answers):

- **Groundedness pass rate: 11/12 (91.7%)**
- **Route distribution:** 8 simple, 3 complex, 1 casual
- **Retry rate:** 1/12 questions required a retrieval retry

The single failure was a question with no answer in the source documents ("what is the capital of France?") — the system correctly recognized it lacked grounding and responded "I can't answer that" rather than hallucinating a response. This is treated as a pass condition for the groundedness mechanism, even though it's logged as a "fail" in the raw pass/fail count, since the system behaved exactly as intended.

## A real debugging example

Initial testing surfaced a retrieval quality issue: asking "what is a Savonius turbine?" returned an accurate answer, but the retrieved chunks shown in the transparency panel were about unrelated adjacent topics (Darrieus turbine disadvantages, wind power formulas) rather than the actual Savonius definition. The model was filling the gap from its own background knowledge rather than the retrieved context.

**Diagnosis:** with 500-word chunks, section headers were frequently separated from their body text during chunking, so the "Savonius" section header ended up in a different chunk than its description.

**Fix:** reduced chunk size to 250 words with a proportionally larger overlap (75 words), keeping headers closer to their content. Re-ingested and re-tested — the correct Savonius definition chunk then appeared as the second-highest-ranked result (relevance 0.63), directly reflected in the generated answer.

## A second real debugging example: shared state in production

During CI setup, the eval suite started passing at 100% — but every answer said "there is no information in the given context," even for questions with obvious answers in the source PDFs. Deeper investigation (adding chunk-level debug logging to the retrieval node) revealed the actual retrieved chunks were always the same 2 chunks from an unrelated resume PDF, regardless of the query topic.

**Root cause:** while testing the Streamlit file-upload feature, a test upload with "Replace existing documents" checked had wiped the shared Pinecone index and BM25 index, replacing 115 chunks of renewable-energy content with just 2 resume chunks. Since Pinecone is a shared cloud resource, this silently broke the local app, the deployed app, and CI simultaneously.

This is a good illustration of why groundedness checks alone aren't sufficient for eval design: the system was technically "passing" by honestly reporting it found nothing, which masked a real underlying data problem. The fix had two parts: restoring the correct document set, and adding a confirmation guardrail in the UI so a destructive "replace all documents" action requires explicit double-confirmation before running.

## Live demo

**[documind-ansh.streamlit.app](https://documind-ansh.streamlit.app)**

Upload a PDF directly through the sidebar and ask questions about it, or try the pre-loaded renewable energy notes with a question like "what is a Savonius turbine?".

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Groq, Cohere, Pinecone, and LangSmith API keys (all free tier).
3. Create a Pinecone index named `documind-index`, dimension `1024`, metric `cosine`, model `llama-text-embed-v2`.
4. Add PDFs to `data/`.
5. Build the index: `python ingestion/build_index.py`
6. Run the app: `streamlit run app/app.py`

To run the eval suite: `python -m eval.run_eval`

## Known limitations / future improvements

- Retry cap is set to 1 (not the 3 typically used in production) to stay within free-tier LLM rate limits — a production deployment would raise this.
- Chunking is word-count-based rather than semantic/structure-aware; a header-aware or sentence-boundary-aware chunker would likely improve retrieval precision further.
- No persistent conversation memory across sessions — each question is handled independently.
- CI (GitHub Actions) gates code quality but does not block Streamlit Cloud deployment, since Streamlit Cloud auto-deploys on every push independently of CI results.