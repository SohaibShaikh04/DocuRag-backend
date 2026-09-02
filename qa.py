"""
qa.py — Production-grade RAG retrieval and generation.

v3 fixes:
  • SCORE_THRESHOLD raised to 0.75 (was 0.30 — was filtering out 95% of valid chunks)
  • TOP_K raised to 10 for better multi-page coverage
  • stream_answer() accepts pre-computed contexts → no double retrieval
  • retrieve_context() falls back gracefully, never returns empty when data exists
"""

import hashlib
import logging
import os
from functools import lru_cache
from typing import AsyncGenerator, List, Optional, Tuple

from groq import Groq

from ingest import _get_embedding_model, embed_texts, get_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (all overridable via .env)
# ---------------------------------------------------------------------------

GROQ_MODEL      = os.getenv("GROQ_MODEL",      "groq/compound-mini")
TOP_K           = int(os.getenv("RAG_TOP_K",   "10"))
# Cosine DISTANCE threshold: lower = more similar.
# 0.75 means we accept chunks with similarity >= 0.25 — permissive enough
# for multi-page docs while still excluding totally unrelated content.
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE", "0.75"))
MAX_TOKENS      = int(os.getenv("RAG_MAX_TOK", "1024"))

# ---------------------------------------------------------------------------
# Groq singleton
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Cached query embedding
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _embed_query_cached(q_hash: str, question: str) -> Tuple[float, ...]:
    """Cache the last 256 query embeddings (same question → zero cost)."""
    return tuple(embed_texts([question])[0])


def embed_query(question: str) -> List[float]:
    q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()
    return list(_embed_query_cached(q_hash, question.strip()))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    collection_id: str,
    question:      str,
    top_k:         int = TOP_K,
) -> List[dict]:
    """
    Retrieve the most relevant chunks from ChromaDB.

    Strategy:
      1. Embed question (cached)
      2. Fetch top_k candidates from HNSW index
      3. Apply lenient distance filter (SCORE_THRESHOLD = 0.75)
         → keeps chunks with cosine similarity ≥ 0.25
      4. Deduplicate near-identical chunks from the same page
      5. Sort by page number so the LLM reads context in document order

    Always returns at least the best-matching chunk even if all fail the filter.
    """
    query_vec  = embed_query(question)
    collection = get_collection(collection_id)
    n_docs     = collection.count()

    if n_docs == 0:
        logger.warning("Collection %s is empty.", collection_id)
        return []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, n_docs),
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Build candidate list
    candidates: List[dict] = []
    for doc, meta, dist in zip(docs, metas, distances):
        candidates.append({
            "text":   doc,
            "page":   meta.get("page", 0),
            "source": meta.get("source", ""),
            "score":  round(1.0 - dist, 4),   # convert distance → similarity
            "dist":   dist,
        })

    # Filter by distance threshold (lenient)
    filtered = [c for c in candidates if c["dist"] <= SCORE_THRESHOLD]

    # Guarantee at least one result
    if not filtered:
        filtered = candidates[:1]

    # Deduplicate (same page + matching first 100 chars)
    seen:   set        = set()
    unique: List[dict] = []
    for c in filtered:
        key = (c["page"], c["text"][:100].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Sort by page for coherent reading order
    unique.sort(key=lambda x: x["page"])

    logger.info(
        "Query '%s…' → %d/%d chunks kept (pages: %s)",
        question[:50],
        len(unique),
        len(docs),
        sorted({c["page"] for c in unique}),
    )
    return unique


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise document Q&A assistant. Your ONLY job is to answer the question using the provided context.

STRICT OUTPUT RULES — follow exactly:
- Start your reply DIRECTLY with the answer. Zero preamble.
- Do NOT write "Based on the document", "According to the context", "Great question", or any intro phrase.
- Cite page numbers inline like this: (Page 3) or (Pages 2, 5).
- Use bullet points only when the answer is a list of multiple items.
- If the answer is a single fact, write one sentence.
- If the information is not in the context, reply ONLY with: "Not found in the document."
- Never repeat the question. Never explain your reasoning. Never add closing remarks.
"""


# ---------------------------------------------------------------------------
# Think-block stripping (Qwen3 safety net)
# ---------------------------------------------------------------------------

import re as _re

def _strip_think(text: str) -> str:
    """Remove any <think>...</think> block from a complete response."""
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    return text.strip()


async def _filter_think_stream(async_gen):
    """
    Remove <think>...</think> from a streaming response.

    Buffers until the opening tag is confirmed or ruled out, then:
      • If inside a think block — silently discard until </think>
      • Once past the think block — yield all deltas immediately
    """
    OPEN  = "<think>"
    CLOSE = "</think>"

    buffer   = ""
    thinking = None   # None=undecided  True=in-think  False=past-think

    async for chunk in async_gen:
        if thinking is False:
            yield chunk
            continue

        buffer += chunk

        if thinking is None:
            if len(buffer) >= len(OPEN):
                if buffer.startswith(OPEN):
                    thinking = True
                else:
                    # Not a think block — flush and pass through
                    thinking = False
                    yield buffer
                    buffer = ""

        if thinking is True and CLOSE in buffer:
            end_idx   = buffer.index(CLOSE) + len(CLOSE)
            remaining = buffer[end_idx:].lstrip("\n")
            buffer    = ""
            thinking  = False
            if remaining:
                yield remaining

    # Flush any residual buffer (e.g. very short response)
    if buffer:
        if thinking is not True:
            yield buffer




def build_messages(contexts: List[dict], question: str) -> List[dict]:
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(
            f"[Page {c['page']}]\n{c['text']}"
        )

    context_str  = "\n\n---\n\n".join(blocks)
    user_content = (
        f"DOCUMENT CONTEXT:\n\n{context_str}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (direct, cite pages, no preamble):"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------

def answer_question(collection_id: str, question: str) -> dict:
    contexts = retrieve_context(collection_id, question)
    if not contexts:
        return {"answer": "I could not find relevant information in the document.", "sources": []}

    completion = _get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=build_messages(contexts, question),
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )

    raw_answer = completion.choices[0].message.content or ""
    return {
        "answer":  _strip_think(raw_answer),   # strip any <think> block
        "sources": [{"page": c["page"], "text": c["text"]} for c in contexts],

    }


# ---------------------------------------------------------------------------
# Streaming  (accepts pre-computed contexts to avoid double retrieval)
# ---------------------------------------------------------------------------

async def stream_answer(
    collection_id:       str,
    question:            str,
    precomputed_contexts: Optional[List[dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the answer token-by-token.

    Pass ``precomputed_contexts`` (from main.py's prior retrieve call) to
    avoid querying ChromaDB a second time and to guarantee sources + answer
    use identical context.
    """
    contexts = precomputed_contexts if precomputed_contexts is not None \
               else retrieve_context(collection_id, question)

    if not contexts:
        yield "I could not find relevant information in the document."
        return

    stream = _get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=build_messages(contexts, question),
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    async def _raw():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # Pipe through think-block filter before yielding to the client
    async for token in _filter_think_stream(_raw()):
        yield token


def get_contexts_for_question(collection_id: str, question: str) -> List[dict]:
    """Thin wrapper for use with run_in_executor in main.py."""
    return retrieve_context(collection_id, question)
