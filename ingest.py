"""
ingest.py — High-performance PDF ingestion pipeline.

v3 fixes:
  • Serial page extraction (PyMuPDF is not thread-safe with shared doc objects)
  • Confirmed working with ChromaDB 0.5.0 HNSW params (only hnsw:space, hnsw:M)
  • Reliable chunking that preserves every page's content
"""

import hashlib
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

CHUNK_SIZE         = 1_200   # chars — rich context per chunk
CHUNK_OVERLAP      = 150     # overlap so answers don't split across chunks
EMBED_BATCH_SIZE   = 64      # saturates CPU BLAS without OOM
CHROMA_BATCH_SIZE  = 100     # safely under ChromaDB hard limit of 166
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client:   Optional[chromadb.ClientAPI]  = None
_executor:        Optional[ThreadPoolExecutor]   = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformer model…")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Warm-up pass
        _embedding_model.encode(["warm-up"], batch_size=1, normalize_embeddings=True)
        logger.info("Embedding model ready.")
    return _embedding_model


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_thread_pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ingest"
        )
    return _executor


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_CTRL_RE         = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE  = re.compile(r"[ \t]{3,}")
_MULTI_NL_RE     = re.compile(r"\n{4,}")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub("  ", text)
    text = _MULTI_NL_RE.sub("\n\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF parsing  (serial — PyMuPDF is not thread-safe with shared doc objects)
# ---------------------------------------------------------------------------

def parse_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extract and clean text from every page of a PDF.

    Returns a list of (1-indexed page number, cleaned text) tuples,
    skipping pages with fewer than 20 characters of usable text.
    """
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)

    for i in range(total):
        try:
            page = doc[i]
            raw  = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            text = _clean(raw)
            if len(text) >= 20:
                pages.append((i + 1, text))
        except Exception as exc:
            logger.warning("Skipping page %d — extraction error: %s", i + 1, exc)

    doc.close()
    logger.info("Extracted text from %d / %d pages.", len(pages), total)
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int  = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """
    Split every page into overlapping text chunks, preserving page metadata.
    Tiny fragments (<30 chars) are skipped.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[dict] = []
    for page_num, page_text in pages:
        splits = splitter.split_text(page_text)
        for idx, split in enumerate(splits):
            text = split.strip()
            if len(text) >= 30:
                chunks.append({"text": text, "page": page_num, "chunk_index": idx})

    logger.info("Created %d chunks from %d pages.", len(chunks), len(pages))
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using all-MiniLM-L6-v2 with:
      • batch_size=64          — saturates CPU BLAS
      • normalize_embeddings=True — cosine sim == dot product (faster retrieval)
    """
    model = _get_embedding_model()
    embs  = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collection_name_for(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()


def ingest_pdf(pdf_bytes: bytes, filename: str) -> Dict:
    """
    Full pipeline (synchronous — call via run_in_executor from async code):
      parse → chunk → embed → store in ChromaDB

    Returns metadata dict: collection_id, num_pages, num_chunks, filename.
    """
    import time
    t0 = time.perf_counter()

    col_name = collection_name_for(filename)

    # 1. Parse — serial, reliable
    pages = parse_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError("The uploaded PDF contains no extractable text.")

    # 2. Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("Could not extract any text chunks from the PDF.")

    texts = [c["text"] for c in chunks]

    # 3. Embed
    t1         = time.perf_counter()
    embeddings = embed_texts(texts)
    logger.info("Embedded %d chunks in %.2fs", len(chunks), time.perf_counter() - t1)

    # 4. Store in ChromaDB
    client = _get_chroma_client()
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M":     32,
        },
    )

    ids       = [f"{col_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"page": c["page"], "source": filename, "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ingested '%s' — %d pages, %d chunks in %.2fs",
        filename, len(pages), len(chunks), elapsed,
    )
    return {
        "collection_id": col_name,
        "num_pages":     len(pages),
        "num_chunks":    len(chunks),
        "filename":      filename,
    }


def get_collection(collection_id: str) -> chromadb.Collection:
    return _get_chroma_client().get_collection(collection_id)
