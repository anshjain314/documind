"""
graph/retriever.py

Hybrid retrieval: combines Pinecone's semantic search with a local BM25
keyword search, fuses the two ranked lists (reciprocal rank fusion), then
reranks the fused candidates with Cohere's reranker to get the final,
most-relevant chunks.
"""

import os
import pickle
from pathlib import Path

import cohere
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

BM25_INDEX_PATH = Path("ingestion/bm25_index.pkl")

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_index = _pc.Index(os.environ["PINECONE_INDEX_NAME"])
_co = cohere.Client(os.environ["COHERE_API_KEY"])

with open(BM25_INDEX_PATH, "rb") as f:
    _bm25_data = pickle.load(f)
    _bm25 = _bm25_data["bm25"]
    _bm25_chunks = _bm25_data["chunks"]


def _semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Search Pinecone (embeds the query automatically via llama-text-embed-v2)."""
    results = _index.search(
        namespace="documind",
        query={"inputs": {"text": query}, "top_k": top_k},
    )
    hits = results.result.hits if hasattr(results, "result") else results.get("result", {}).get("hits", [])

    parsed = []
    for h in hits:
        # Handle both object-style (attribute access) and dict-style results,
        # since different Pinecone SDK versions return different shapes.
        h_id = getattr(h, "_id", None) or getattr(h, "id", None) or (h.get("_id") if hasattr(h, "get") else None)
        h_score = getattr(h, "_score", None) or getattr(h, "score", None) or (h.get("_score") if hasattr(h, "get") else 0.0)
        h_fields = getattr(h, "fields", None) or (h.get("fields") if hasattr(h, "get") else {})

        parsed.append({
            "id": h_id,
            "text": h_fields.get("text", "") if hasattr(h_fields, "get") else getattr(h_fields, "text", ""),
            "source": h_fields.get("source", "") if hasattr(h_fields, "get") else getattr(h_fields, "source", ""),
            "score": h_score or 0.0,
        })
    return parsed


def _keyword_search(query: str, top_k: int = 10) -> list[dict]:
    """Search the local BM25 index."""
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)
    ranked = sorted(
        zip(_bm25_chunks, scores), key=lambda x: x[1], reverse=True
    )[:top_k]
    return [
        {"id": c["id"], "text": c["text"], "source": c["source"], "score": s}
        for c, s in ranked
        if s > 0
    ]


def _reciprocal_rank_fusion(
    semantic_results: list[dict], keyword_results: list[dict], k: int = 60
) -> list[dict]:
    """Combine two ranked lists into one fused ranking."""
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(semantic_results):
        scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (k + rank + 1)
        chunk_lookup[chunk["id"]] = chunk

    for rank, chunk in enumerate(keyword_results):
        scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (k + rank + 1)
        chunk_lookup[chunk["id"]] = chunk

    fused_ids = sorted(scores, key=scores.get, reverse=True)
    return [chunk_lookup[i] for i in fused_ids]


def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full hybrid retrieval pipeline: semantic + keyword search -> fusion -> rerank.
    Returns the final top_k chunks, reranked by Cohere for relevance.
    """
    semantic_results = _semantic_search(query, top_k=10)
    keyword_results = _keyword_search(query, top_k=10)
    fused = _reciprocal_rank_fusion(semantic_results, keyword_results)

    if not fused:
        return []

    # Cohere rerank on the fused candidate list
    docs = [c["text"] for c in fused]
    rerank_results = _co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=min(top_k, len(docs)),
    )

    reranked = []
    for r in rerank_results.results:
        chunk = fused[r.index]
        chunk["rerank_score"] = r.relevance_score
        reranked.append(chunk)

    return reranked