"""Эвристики очистки Markdown от повторяющихся шапок."""

from __future__ import annotations

import re
from collections import Counter
from typing import Tuple

from .config import load_header_cfg


def detect_and_remove_repeating_header(md_text: str) -> Tuple[str, int]:
    """Удаляет повторяющиеся верхние строки абзацев по всему документу."""
    if not md_text:
        return md_text, 0
    blocks = [b for b in re.split(r"\n\s*\n", md_text) if b.strip()]
    first_lines = []
    for b in blocks:
        line = b.splitlines()[0].strip()
        norm = re.sub(r"[\W_0-9]+", " ", line).strip().upper()
        first_lines.append(norm)
    freq = Counter(first_lines)
    if not freq:
        return md_text, 0
    cfg = load_header_cfg()
    min_repeats = int(cfg.get("min_repeats", 3))
    min_length = int(cfg.get("min_length", 4))
    up_thresh = float(cfg.get("uppercase_ratio", 0.6))
    org_keywords = tuple(cfg.get("keywords", []))
    candidates = set()
    for norm, cnt in freq.items():
        if cnt < min_repeats or len(norm) < min_length:
            continue
        has_kw = any(k in norm for k in org_keywords)
        up_ratio = sum(
            1 for ch in norm if "A" <= ch <= "Z" or "А" <= ch <= "Я"
        ) / max(1, len(norm.replace(" ", "")))
        if has_kw or up_ratio > up_thresh:
            candidates.add(norm)
    if not candidates:
        return md_text, 0
    removed = 0
    new_blocks = []
    for b in blocks:
        lines = b.splitlines()
        if not lines:
            new_blocks.append(b)
            continue
        norm0 = re.sub(r"[\W_0-9]+", " ", lines[0].strip()).strip().upper()
        if norm0 in candidates:
            removed += 1
            lines = lines[1:]
        new_blocks.append("\n".join(lines).strip())
    cleaned = "\n\n".join([blk for blk in new_blocks if blk])
    return cleaned, removed


def remove_first_page_header(md_text: str) -> Tuple[str, int]:
    """Удаляет многосрочную «шапку» только вверху первой страницы."""
    if not md_text:
        return md_text, 0
    cfg = load_header_cfg()
    fp = cfg.get("first_page", {}) or {}
    if not bool(fp.get("enable", True)):
        return md_text, 0
    lines_limit = int(fp.get("lines_limit", 20))
    max_block_lines = int(fp.get("max_block_lines", 6))
    up_thresh = float(fp.get("uppercase_ratio", cfg.get("uppercase_ratio", 0.6)))
    org_keywords = tuple(fp.get("keywords", cfg.get("keywords", [])))

    lines = md_text.splitlines()
    if not lines:
        return md_text, 0

    top: list[str] = []
    for ln in lines[:lines_limit]:
        if ln.strip() == "":
            break
        top.append(ln)
        if len(top) >= max_block_lines:
            break
    if len(top) < 2:
        return md_text, 0

    has_kw = any(any(k in ln.upper() for k in org_keywords) for ln in top)

    def up_ratio(s: str) -> float:
        norm = re.sub(r"[\W_0-9]+", "", s)
        if not norm:
            return 0.0
        ups = sum(1 for ch in norm if "A" <= ch <= "Z" or "А" <= ch <= "Я")
        return ups / len(norm)

    avg_up = sum(up_ratio(ln) for ln in top) / max(1, len(top))
    if has_kw or avg_up > up_thresh:
        removed = len(top)
        rest = "\n".join(lines[removed:])
        return rest.lstrip("\n"), removed
    return md_text, 0


# Обратная совместимость со старыми импортами.
_detect_and_remove_repeating_header = detect_and_remove_repeating_header
_remove_first_page_header = remove_first_page_header
