"""Гибридный алгоритм чанкинга Markdown-текста."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .schemas import ChunkItem


def split_code_fences(text: str) -> List[Tuple[str, bool]]:
    """Разбивает текст на блоки: обычные и блоки кода (```...```))."""
    parts: list[tuple[str, bool]] = []
    fence_re = re.compile(r"(^```[\s\S]*?^```\s*$)", re.MULTILINE)
    last = 0
    for match in fence_re.finditer(text):
        if match.start() > last:
            parts.append((text[last : match.start()], False))
        parts.append((match.group(1), True))
        last = match.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts


def split_paragraphs(block: str) -> List[str]:
    """Делит блок на параграфы, устраняя лишние пустые строки."""
    paras = re.split(r"\n\s*\n", block.strip())
    out: list[str] = []
    for paragraph in paras:
        lines = [ln for ln in paragraph.splitlines() if ln.strip() != ""]
        if lines:
            out.append("\n".join(lines))
    return out


def split_sentences(text: str) -> List[str]:
    """Простая нарезка текста по предложениям."""
    parts = re.split(r"(?<=[\.!?…])\s+(?=[A-ZА-ЯЁ])", text)
    return [part for part in parts if part]


def hybrid_chunk(text: str, max_chars: int, overlap: int, preserve_md: bool) -> list[ChunkItem]:
    """Гибридный чанкер, учитывающий код, параграфы, предложения и overlap."""
    text = text or ""
    blocks = split_code_fences(text)
    chunks: list[ChunkItem] = []
    buf = ""
    pos = 0

    def flush(with_overlap: bool = True) -> None:
        nonlocal buf, pos
        if buf.strip() == "":
            return
        start = pos
        end = pos + len(buf)
        idx = len(chunks)
        chunks.append(ChunkItem(index=idx, start=start, end=end, text=buf))
        if with_overlap and overlap > 0 and len(buf) > overlap:
            buf = buf[-overlap:]
            pos = end - overlap
        else:
            buf = ""
            pos = end

    for block, is_code in blocks:
        if is_code:
            if buf and len(buf) + len(block) + 2 > max_chars:
                flush()
            buf = f"{buf}\n\n{block}" if buf else block
            flush()
            continue

        for paragraph in split_paragraphs(block):
            is_heading = bool(re.match(r"^\s*#{1,6}\s+", paragraph)) if preserve_md else False
            if is_heading:
                if buf:
                    flush()
                if len(paragraph) >= max_chars:
                    buf = paragraph[:max_chars]
                    flush()
                    buf = paragraph[max_chars:]
                    flush()
                else:
                    buf = paragraph
                continue

            if len(paragraph) + (2 if buf else 0) <= max_chars - len(buf):
                buf = f"{buf}\n\n{paragraph}" if buf else paragraph
                continue

            sents = split_sentences(paragraph)
            cur = ""
            for sent in sents:
                candidate = sent if cur == "" else (cur + " " + sent)
                if len(candidate) <= max_chars - len(buf) - (2 if buf else 0):
                    cur = candidate
                else:
                    if cur:
                        buf = f"{buf}\n\n{cur}" if buf else cur
                        flush()
                        cur = ""
                    while len(sent) > max_chars:
                        part = sent[:max_chars]
                        sent = sent[max_chars:]
                        buf = part
                        flush()
                    cur = sent
            if cur:
                if len(cur) + (2 if buf else 0) > max_chars - len(buf):
                    flush()
                buf = f"{buf}\n\n{cur}" if buf else cur
                flush()

    if buf:
        flush(with_overlap=False)

    return chunks
