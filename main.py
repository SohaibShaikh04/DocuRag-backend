"""
main.py — FastAPI application.

v3 fixes:
  • stream-chat passes pre-computed contexts to stream_answer (no double retrieval)
  • Structured logging with timings
  • ingest runs in thread-pool (non-blocking event loop)
"""

import json
import logging
import os
from asyncio import get_event_loop
from contextlib import asynccontextmanager
from functools import partial
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

from ingest import _get_embedding_model, get_thread_pool, ingest_pdf
from qa import answer_question, get_contexts_for_question, stream_answer

# ---------------------------------------------------------------------------
# Lifespan — pre-warm model at startup so first upload is not cold
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-warming embedding model…")
    loop = get_event_loop()
    await loop.run_in_executor(get_thread_pool(), _get_embedding_model)
    logger.info("Server ready.")
    yield
    get_thread_pool().shutdown(wait=False)


app = FastAPI(
    title="Chat with PDF — RAG API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    collection_id: str
    question: str

class SourceChunk(BaseModel):
    page: int
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class UploadResponse(BaseModel):
    collection_id: str
    filename: str
    num_pages: int
    num_chunks: int

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    from qa import GROQ_MODEL
    return {"status": "ok", "model": GROQ_MODEL, "version": "3.0.0"}


@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF and run the full ingestion pipeline in a thread-pool.
    Returns collection metadata (id, pages, chunks).
    """
    fname = file.filename or "document.pdf"
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(pdf_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            413,
            f"File exceeds 20 MB limit ({len(pdf_bytes)/1_048_576:.1f} MB).",
        )

    try:
        loop   = get_event_loop()
        result = await loop.run_in_executor(
            get_thread_pool(),
            partial(ingest_pdf, pdf_bytes, fname),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for '%s'", fname)
        raise HTTPException(500, f"Ingestion failed: {exc}")

    return UploadResponse(
        collection_id=result["collection_id"],
        filename=result["filename"],
        num_pages=result["num_pages"],
        num_chunks=result["num_chunks"],
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint."""
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    try:
        loop   = get_event_loop()
        result = await loop.run_in_executor(
            get_thread_pool(),
            partial(answer_question, req.collection_id, req.question),
        )
    except Exception as exc:
        _handle_groq_error(exc)

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.get("/stream-chat", tags=["Chat"])
async def stream_chat(collection_id: str, question: str):
    """
    Streaming SSE endpoint.

    Event flow:
      sources → JSON array of {page, text} chunks (sent first, before streaming)
      data    → incremental answer token
      done    → stream complete
      error   → error string (or "rate_limit")
    """
    if not question.strip():
        raise HTTPException(400, "Question must not be empty.")

    async def event_generator():
        try:
            # ── 1. Retrieve context ONCE (cached embedding + ChromaDB query) ──
            loop     = get_event_loop()
            contexts = await loop.run_in_executor(
                get_thread_pool(),
                partial(get_contexts_for_question, collection_id, question),
            )

            # ── 2. Send sources immediately so frontend can show them ──
            sources_payload = json.dumps(
                [{"page": c["page"], "text": c["text"]} for c in contexts]
            )
            yield {"event": "sources", "data": sources_payload}

            # ── 3. Stream answer — REUSE the contexts, no second retrieval ──
            async for delta in stream_answer(
                collection_id,
                question,
                precomputed_contexts=contexts,   # ← key fix: no double query
            ):
                yield {"event": "data", "data": delta}

            yield {"event": "done", "data": ""}

        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                yield {"event": "error", "data": "rate_limit"}
            else:
                logger.exception("SSE stream error for collection %s", collection_id)
                yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handle_groq_error(exc: Exception):
    err = str(exc).lower()
    if "rate_limit" in err or "429" in err:
        raise HTTPException(429, "Rate limit reached. Please wait a moment.")
    raise HTTPException(500, f"Chat failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
