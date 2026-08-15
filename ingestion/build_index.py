"""
ingestion/build_index.py

Reads all PDFs from /data, splits them into overlapping chunks,
pushes them to Pinecone (which embeds them automatically via
llama-text-embed-v2 integrated inference), and builds a local
BM25 keyword index for hybrid search later.

Run this from the project root:
    python ingestion/build_index.py
"""

import os
import re
import pickle
from pathlib import Path

from pypdf import PdfReader
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
BM25_INDEX_PATH = Path("ingestion/bm25_index.pkl")
CHUNK_SIZE = 250       # words per chunk (smaller keeps headers closer to their content)
CHUNK_OVERLAP = 75      # words of overlap between chunks


def load_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a single PDF."""
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    # collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")

    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        if chunk_str.strip():
            chunks.append({
                "id": f"{source}-{chunk_id}",
                "text": chunk_str,
                "source": source,
            })
            chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_pdfs(pdf_paths: list[Path], clear_existing: bool = True, progress_callback=None) -> int:
    """
    Core ingestion logic: reads PDFs, chunks them, pushes to Pinecone, and
    rebuilds the local BM25 index. Used by both the CLI script and the
    Streamlit upload feature.

    Args:
        pdf_paths: list of PDF file paths to ingest
        clear_existing: if True, wipes the Pinecone namespace and BM25 index
                         first (fresh build). If False, adds to what's there
                         (used when a user uploads an additional file via the UI).
        progress_callback: optional function(str) called with status updates,
                            so the Streamlit UI can show live progress.

    Returns:
        total number of chunks ingested
    """
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    all_chunks = []
    for pdf_path in pdf_paths:
        log(f"Reading {pdf_path.name} ...")
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text, source=pdf_path.stem)
        log(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    log(f"Total chunks: {len(all_chunks)}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    if clear_existing:
        log("Clearing previous chunks from Pinecone namespace...")
        try:
            index.delete(delete_all=True, namespace="documind")
        except Exception as e:
            log(f"  (namespace was likely already empty: {e})")

    log("Upserting to Pinecone (this embeds automatically via llama-text-embed-v2)...")
    records = [
        {"_id": c["id"], "text": c["text"], "source": c["source"]}
        for c in all_chunks
    ]

    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert_records(namespace="documind", records=batch)
        log(f"  Upserted {i + len(batch)}/{len(records)}")

    log("Pinecone upload complete.")

    # --- Build/update local BM25 keyword index ---
    log("Building BM25 keyword index...")

    existing_chunks = []
    if not clear_existing and BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            existing_data = pickle.load(f)
            existing_chunks = existing_data.get("chunks", [])

    combined_chunks = existing_chunks + all_chunks
    tokenized_corpus = [c["text"].lower().split() for c in combined_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": combined_chunks}, f)

    log(f"BM25 index saved to {BM25_INDEX_PATH}")
    log("Ingestion complete.")

    return len(all_chunks)


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR.resolve()}. Add some and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
    ingest_pdfs(pdf_files, clear_existing=True)


if __name__ == "__main__":
    main()