"""Маршруты чанкинга текста."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..chunking import hybrid_chunk
from ..schemas import ChunkRequest, ChunkResponse

router = APIRouter()


@router.post("/chunk", response_model=ChunkResponse)
async def chunk_text(req: ChunkRequest) -> ChunkResponse:
    """Чанкинг текста гибридным алгоритмом."""
    if req.max_chars <= 0:
        raise HTTPException(400, detail="max_chars must be > 0")
    if req.overlap < 0:
        raise HTTPException(400, detail="overlap must be >= 0")

    chunks = hybrid_chunk(
        text=req.text,
        max_chars=req.max_chars,
        overlap=req.overlap,
        preserve_md=req.preserve_markdown,
    )
    return ChunkResponse(
        chunks=chunks,
        meta={
            "strategy": "hybrid",
            "count": len(chunks),
            "max_chars": req.max_chars,
            "overlap": req.overlap,
        },
    )
