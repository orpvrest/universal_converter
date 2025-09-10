"""Сервис FastAPI на базе Docling.

Сервис конвертирует офисные документы, PDF и изображения в структурированный
вид (Markdown и/или JSON) с использованием Docling. Поддерживается
автоопределение формата, при необходимости — конвертация устаревших форматов
через LibreOffice (.doc/.xls/.ppt → OOXML) и выбор стратегии OCR (Tesseract)
для «изобразительных» входов (без перебора PSM).

Переменные окружения:
    UVICORN_HOST (str): Хост для uvicorn. По умолчанию: 0.0.0.0
    UVICORN_PORT (int): Порт для uvicorn. По умолчанию: 8088
    OMP_NUM_THREADS (int): Кол-во потоков OpenMP. По умолчанию: 4
    DOCLING_ARTIFACTS_PATH (str): Папка с моделями Docling.
    DOCLING_TABLE_MODE (str): Режим таблиц, напр. "ACCURATE".
    DOCLING_FORCE_OCR (bool): Принудительный OCR (базовое значение).
        По умолчанию: true
    DOCLING_LANGS (str): Языки OCR через запятую, напр. "rus,eng".
    MAX_FILE_SIZE_MB (int): Максимальный размер файла в МБ. По умолчанию: 80

Основные эндпоинты:
    GET /health
        Простой чек доступности.

    POST /convert
        Универсальный эндпоинт со стратегией по типу файла:
    - .doc/.xls/.ppt → LibreOffice → OOXML → Docling (без OCR)
    - PDF/изображения → без OCR, затем принудительный OCR; выбор лучшего
        - OOXML/HTML/MD/CSV → Docling (без OCR)
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import tempfile
import re
from typing import Literal, Optional
import shutil
from collections import Counter
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from typing import Any

app = FastAPI(title="Docling microservice", version="3.2.0")

DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH", None)
DEFAULT_TABLE_MODE = os.getenv("DOCLING_TABLE_MODE", "ACCURATE").upper()
DEFAULT_FORCE_OCR = os.getenv("DOCLING_FORCE_OCR", "true").lower() == "true"
DEFAULT_LANGS = [
    x.strip()
    for x in os.getenv("DOCLING_LANGS", "rus,eng").split(",")
    if x.strip()
]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "80"))
HEADER_CFG_PATH = os.getenv(
    "HEADER_FILTER_CONFIG",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "header_filter.json",
        )
    ),
)
_HEADER_CFG = None  # lazy-loaded dict
_MODELS_READY = False  # lazy init flag for docling models


def _load_header_cfg() -> dict:
    global _HEADER_CFG
    if _HEADER_CFG is not None:
        return _HEADER_CFG
    try:
        with open(HEADER_CFG_PATH, "r", encoding="utf-8") as f:
            _HEADER_CFG = json.load(f)
    except Exception:
        # Defaults if config missing/broken
        _HEADER_CFG = {
            "min_repeats": 3,
            "min_length": 4,
            "uppercase_ratio": 0.6,
            "keywords": [
                "ООО", "АО", "ПАО", "ФГУП", "ФГБУ", "МИНИСТЕРСТВО",
                "АДМИНИСТРАЦИЯ", "КОМПАНИЯ", "КОРПОРАЦИЯ", "УНИВЕРСИТЕТ",
                "ИНСТИТУТ", "ФЕДЕРАЛЬНОЕ", "ГОСУДАРСТВЕННОЕ", "ОБЩЕСТВО",
                "ОРГАНИЗАЦИЯ", "РОССИЯ", "РОССИЙСКАЯ"
            ],
        }
    return _HEADER_CFG


# Ленивая проверка/загрузка моделей

def ensure_models() -> None:
    global _MODELS_READY
    if _MODELS_READY:
        return
    if not DOCLING_ARTIFACTS_PATH:
        # Пусть Docling использует дефолтный кэш
        _MODELS_READY = True
        return
    try:
        from pathlib import Path
        import importlib

        mdl = importlib.import_module("docling.utils.model_downloader")
        download_models = getattr(mdl, "download_models")

        root = Path(DOCLING_ARTIFACTS_PATH)
        root.mkdir(parents=True, exist_ok=True)

        # Убедимся, что модели есть (или скачаем)
        found_any = list(root.rglob("model.safetensors"))
        if not found_any:
            download_models(
                output_dir=root,
                progress=False,
                with_layout=True,
                with_tableformer=True,
                with_code_formula=True,
                with_picture_classifier=True,
                with_easyocr=True,
            )
            found_any = list(root.rglob("model.safetensors"))

    # Docling (LayoutPredictor) может ожидать safetensors
    # по корневому пути.
        # Если в корне нет файла, но он есть в подпапках — положим симлинк.
        root_st = root / "model.safetensors"
        if not root_st.exists() and found_any:
            target = found_any[0]
            try:
                # Симлинк предпочтительнее, но если FS не поддерживает —
                # копируем
                root_st.symlink_to(target)
            except Exception:
                try:
                    shutil.copyfile(target, root_st)
                except Exception:
                    pass

        # Кроме safetensors, Docling ожидает ряд файлов прямо в корне
        # каталога артефактов. Если они лежат глубже — положим симлинки/копии.
        model_dir = None
        # Ищем рядом с найденным model.safetensors папку с config/preprocessor
        for st in found_any:
            cand = st.parent
            if (cand / "preprocessor_config.json").exists() or (
                cand / "config.json"
            ).exists():
                model_dir = cand
                break
        # Если не нашли по соседству, попробуем глобально по дереву артефактов
        if model_dir is None:
            try:
                pc = next(root.rglob("preprocessor_config.json"), None)
            except Exception:
                pc = None
            if pc is not None:
                model_dir = pc.parent
        if model_dir is None and found_any:
            model_dir = found_any[0].parent

        if model_dir is not None:
            candidates = [
                "preprocessor_config.json",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ]
            for name in candidates:
                src = model_dir / name
                dst = root / name
                if src.exists() and not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except Exception:
                        try:
                            shutil.copyfile(src, dst)
                        except Exception:
                            pass

        _MODELS_READY = True
    except Exception:
        # Не блокируем: Docling попробет скачать сам в дефолтный кэш
        _MODELS_READY = True


def _langs_join(langs: list[str]) -> str:
    """Формирует строку языков для Tesseract/OCRmyPDF: 'rus+eng'."""
    cleaned = [x for x in (langs or []) if x]
    return "+".join(cleaned) if cleaned else "rus+eng"


def _try_ocrmypdf_preprocess(
    pdf_bytes: bytes, langs: list[str]
) -> Optional[bytes]:
    """Пробует подготовить PDF через OCRmyPDF:
    - deskew/rotate/clean/remove-background/optimize
    - --skip-text (не трогать страницы с уже существующим текстовым слоем)
    Возвращает улучшенный PDF или None, если OCRmyPDF недоступен/ошибся.
    """
    if not shutil.which("ocrmypdf"):
        return None
    in_fd, in_path = tempfile.mkstemp(suffix=".pdf")
    out_fd, out_path = tempfile.mkstemp(suffix=".pdf")
    os.close(in_fd)
    os.close(out_fd)
    try:
        with open(in_path, "wb") as f:
            f.write(pdf_bytes)
        cmd = [
            "ocrmypdf",
            "--skip-text",
            "--rotate-pages",
            "--deskew",
            "--clean",
            "--remove-background",
            "--optimize", "3",
            "--language", _langs_join(langs),
            in_path,
            out_path,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            return None
        # sanity check: выход не пустой и не слишком мал
        data = open(out_path, "rb").read()
        return data if len(data) > 1024 else None
    except Exception:
        return None
    finally:
        try:
            os.remove(in_path)
            os.remove(out_path)
        except Exception:
            pass


def _detect_and_remove_repeating_header(md_text: str) -> tuple[str, int]:
    """Удаляет повторяющуюся «шапку» (верхние строки абзацев)
    по всему документу.
    Эвристики:
      - Считаем первую строку каждого абзаца (разделитель — пустая строка)
      - Нормализуем (убираем пунктуацию/цифры, приводим к VERHNIY REGISTR)
      - Кандидат — встречается >= 3 раз, длина >= 4, и (есть ключевые слова
        организаций/ведомств ИЛИ доля верхнего регистра > 0.6)
      - Удаляем такие строки из начала абзацев
    Возвращает (очищенный_markdown, сколько_раз_удалено).
    """
    if not md_text:
        return md_text, 0
    blocks = [b for b in re.split(r"\n\s*\n", md_text) if b.strip()]
    first_lines = []
    raw_first = []
    for b in blocks:
        line = b.splitlines()[0].strip()
        raw_first.append(line)
        norm = re.sub(r"[\W_0-9]+", " ", line).strip().upper()
        first_lines.append(norm)
    freq = Counter(first_lines)
    if not freq:
        return md_text, 0
    cfg = _load_header_cfg()
    min_repeats = int(cfg.get("min_repeats", 3))
    min_length = int(cfg.get("min_length", 4))
    up_thresh = float(cfg.get("uppercase_ratio", 0.6))
    org_keywords = tuple(cfg.get("keywords", []))
    candidates = set()
    for norm, cnt in freq.items():
        if cnt < min_repeats or len(norm) < min_length:
            continue
        has_kw = any(k in norm for k in org_keywords)
        up_ratio = (
            sum(1 for ch in norm if "A" <= ch <= "Z" or "А" <= ch <= "Я") /
            max(1, len(norm.replace(" ", "")))
        )
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
            lines = lines[1:]  # убрать первую строку как «шапку»
        new_blocks.append("\n".join(lines).strip())
    cleaned = "\n\n".join([blk for blk in new_blocks if blk])
    return cleaned, removed


def _remove_first_page_header(md_text: str) -> tuple[str, int]:
    """Удаляет многосрочную «шапку» только вверху первой страницы/документа.

    Эвристика:
            - Смотрим первые N строк (по умолчанию 20), берём подряд идущие
                непустые строки сверху (до пустой строки), не более M строк
                (по умолчанию 6).
      - Если верхний блок не начинается с Markdown-заголовка (# ...),
        и суммарно удовлетворяет одному из условий:
          • содержит ключевые слова организаций; или
          • средняя доля верхнего регистра по строкам > порога.
        — удаляем этот блок.
    Возвращает (очищенный_markdown, сколько_строк_удалено).
    """
    if not md_text:
        return md_text, 0
    cfg = _load_header_cfg()
    fp = cfg.get("first_page", {}) or {}
    enable = bool(fp.get("enable", True))
    if not enable:
        return md_text, 0
    lines_limit = int(fp.get("lines_limit", 20))
    max_block_lines = int(fp.get("max_block_lines", 6))
    up_thresh = float(
        fp.get("uppercase_ratio", cfg.get("uppercase_ratio", 0.6))
    )
    org_keywords = tuple(fp.get("keywords", cfg.get("keywords", [])))

    lines = md_text.splitlines()
    if not lines:
        return md_text, 0
    # Не трогаем явные заголовки документа
    if lines[0].lstrip().startswith("#"):
        return md_text, 0

    # Собираем верхний непустой блок
    top: list[str] = []
    for i, ln in enumerate(lines[: lines_limit]):
        if ln.strip() == "":
            break
        top.append(ln)
        if len(top) >= max_block_lines:
            break
    if len(top) < 2:
        return md_text, 0

    # Оценим блок
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


# Вспомогательная функция для сборки DocumentConverter под PDF/изображения
def build_pdf_converter(
    force_ocr: bool,
    langs: list[str],
    table_mode: str,
) -> Any:
    """Собирает DocumentConverter для PDF/изображений.

    Args:
        force_ocr: Принудительный полностраничный OCR.
        langs: Языки OCR (например, ["rus", "eng"]).
        table_mode: Режим извлечения таблиц (например, "ACCURATE").

    Returns:
        DocumentConverter, настроенный для обработки PDF/изображений.
    """
    # Убедиться, что модели есть (или будут скачаны в указанный артефакт-путь)
    ensure_models()

    # Lazy import docling internals
    import importlib
    dc_mod = importlib.import_module("docling.document_converter")
    dm_base = importlib.import_module("docling.datamodel.base_models")
    pipe_mod = importlib.import_module("docling.datamodel.pipeline_options")
    DocumentConverter = getattr(dc_mod, "DocumentConverter")
    PdfFormatOption = getattr(dc_mod, "PdfFormatOption")
    InputFormat = getattr(dm_base, "InputFormat")
    PdfPipelineOptions = getattr(pipe_mod, "PdfPipelineOptions")
    TesseractCliOcrOptions = getattr(pipe_mod, "TesseractCliOcrOptions")
    TableFormerMode = getattr(pipe_mod, "TableFormerMode")

    pipe = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS_PATH)

    # When forcing OCR, enable full-page OCR; otherwise disable OCR entirely
    # for a true baseline without OCR.
    if force_ocr:
        pipe.do_ocr = True
        ocr_opts = TesseractCliOcrOptions(
            lang=langs,
            force_full_page_ocr=True,
        )
        pipe.ocr_options = ocr_opts
    else:
        pipe.do_ocr = False

    if table_mode == "ACCURATE":
        pipe.do_table_structure = True
        pipe.table_structure_options.mode = TableFormerMode.ACCURATE
        pipe.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipe),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipe),
        }
    )


class ConvertResponse(BaseModel):
    """Стандартная модель ответа конвертации.

    Attributes:
        content_markdown: Строка Markdown (если запрошено).
        content_json: JSON-структура (если запрошено).
        meta: Метаданные о запросе/обработке.
    """
    content_markdown: Optional[str] = None
    content_json: Optional[dict] = None
    meta: dict


class ChunkItem(BaseModel):
    index: int
    start: int
    end: int
    text: str


class ChunkRequest(BaseModel):
    text: str
    max_chars: int = 2000
    overlap: int = 200
    preserve_markdown: bool = True


class ChunkResponse(BaseModel):
    chunks: list[ChunkItem]
    meta: dict


def _split_code_fences(text: str) -> list[tuple[str, bool]]:
    """Разбивает текст на блоки: обычные и блоки кода (```...```).

    Returns список (block_text, is_code_block).
    """
    parts: list[tuple[str, bool]] = []
    fence_re = re.compile(r"(^```[\s\S]*?^```\s*$)", re.MULTILINE)
    last = 0
    for m in fence_re.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(1), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts


def _split_paragraphs(block: str) -> list[str]:
    # Сначала по двойным переводам строк, затем по одиночным
    paras = re.split(r"\n\s*\n", block.strip())
    out: list[str] = []
    for p in paras:
        lines = [ln for ln in p.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        out.append("\n".join(lines))
    return out


def _split_sentences(text: str) -> list[str]:
    # Простая нарезка по предложениям (латиница+кириллица)
    sents = re.split(r"(?<=[\.!?…])\s+(?=[A-ZА-ЯЁ])", text)
    return [s for s in sents if s]


def hybrid_chunk(
    text: str,
    max_chars: int,
    overlap: int,
    preserve_md: bool,
) -> list[ChunkItem]:
    """Гибридный чанкер: сохраняет MD-заголовки,
    учитывает параграфы/предложения, ограничивает размер чанка и
    добавляет перекрытие overlap.
    """
    text = text or ""
    blocks = _split_code_fences(text)
    chunks: list[ChunkItem] = []
    buf = ""
    pos = 0

    def flush(with_overlap: bool = True):
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
            if buf:
                buf = f"{buf}\n\n{block}" if buf else block
            else:
                buf = block
            flush()
            continue

        paras = _split_paragraphs(block)
        for p in paras:
            is_heading = (
                bool(re.match(r"^\s*#{1,6}\s+", p)) if preserve_md else False
            )
            if is_heading:
                # Заголовок как префикс следующего чанка
                if buf:
                    flush()
                if len(p) >= max_chars:
                    # Очень длинный заголовок — отдадим отдельным чанком
                    buf = p[:max_chars]
                    flush()
                    buf = p[max_chars:]
                    flush()
                else:
                    buf = p
                continue

            # Если параграф помещается
            if len(p) + (2 if buf else 0) <= max_chars - len(buf):
                buf = f"{buf}\n\n{p}" if buf else p
                continue

            # Иначе режем параграф на предложения
            sents = _split_sentences(p)
            cur = ""
            for s in sents:
                add = s if cur == "" else (cur + " " + s)
                if len(add) <= max_chars - len(buf) - (2 if buf else 0):
                    cur = add
                else:
                    if cur:
                        # Финализируем текущую часть параграфа
                        buf = f"{buf}\n\n{cur}" if buf else cur
                        flush()
                        cur = ""
                    # Если предложение само длинное — режем по символам
                    while len(s) > max_chars:
                        part = s[:max_chars]
                        s = s[max_chars:]
                        buf = part
                        flush()
                    cur = s
            if cur:
                if len(cur) + (2 if buf else 0) > max_chars - len(buf):
                    flush()
                buf = f"{buf}\n\n{cur}" if buf else cur
                flush()

    if buf:
        flush(with_overlap=False)

    return chunks


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(req: ChunkRequest) -> ChunkResponse:
    """Чанкинг текста гибридным алгоритмом.

    Вход: JSON с полями text, max_chars, overlap, preserve_markdown.
    Выход: список чанков с индексами и позициями в исходной строке.
    """
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


@app.get("/health")
def health():
    """Эндпоинт проверки доступности.

    Returns:
        dict: {"ok": True}
    """
    return {"ok": True}

# Универсальный эндпоинт: автоопределение + конвертация legacy +
# OCR без перебора PSM


@app.post("/convert", response_model=ConvertResponse)
async def convert_universal(
    file: UploadFile = File(...),
    out: Literal["markdown", "json", "both"] = Form("both"),
    langs: Optional[str] = Form(None),              # "rus,eng" | "auto"
    max_pages: Optional[int] = Form(None),
):
    """Универсальная конвертация документов с авто-стратегией.

    Args:
        file: Загружаемый файл (любой офисный формат, PDF или изображение).
        out: Формат вывода: "markdown", "json" или "both".
        langs: Языки OCR (строка, например "rus,eng"). Если None — берём из
            переменной окружения DOCLING_LANGS.
        max_pages: Ограничение по количеству страниц для обработки (или None).

    Returns:
        fastapi.responses.JSONResponse: Ответ с Markdown/JSON, выбранной
        стратегией и метриками trials.

    Raises:
        HTTPException: Если файл превышает лимит по размеру или все стратегии
        обработки завершились неудачей.
    """
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            413,
            detail=(
                f"File too large: {size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB"
            ),
        )

    filename = file.filename or "upload.bin"
    lower = filename.lower()
    _langs = [
        x.strip()
        for x in (langs or ",".join(DEFAULT_LANGS)).split(",")
        if x.strip()
    ]

    # Local helper to build a Docling stream lazily
    def _mk_stream(name: str, data: bytes):
        import importlib
        dm_base = importlib.import_module("docling.datamodel.base_models")
        DocumentStream = getattr(dm_base, "DocumentStream")
        return DocumentStream(name=name, stream=io.BytesIO(data))

    # helpers
    def is_legacy(name: str) -> bool:
        return (
            name.endswith(".doc")
            or name.endswith(".xls")
            or name.endswith(".ppt")
        )

    def is_pdf(name: str) -> bool:
        return name.endswith(".pdf")

    def is_image(name: str) -> bool:
        return any(
            name.endswith(ext)
            for ext in (
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".webp",
            )
        )

    def is_direct_supported(name: str) -> bool:
        return any(
            name.endswith(ext)
            for ext in (
                ".docx",
                ".pptx",
                ".xlsx",
                ".html",
                ".htm",
                ".md",
                ".csv",
                ".adoc",
                ".asciidoc",
            )
        )

    # Отсечь неподдерживаемые форматы (например .txt)
    if not (
        is_legacy(lower)
        or is_pdf(lower)
        or is_image(lower)
        or is_direct_supported(lower)
    ):
        raise HTTPException(
            415,
            detail=(
                "Unsupported file type. Allowed: legacy (.doc/.xls/.ppt), "
                ".pdf, images, or direct formats "
                "(.docx/.pptx/.xlsx/.html/.md/.csv/.adoc)"
            ),
        )

    # 1) legacy → LibreOffice convert → OOXML bytes
    tmp_in = tmp_out = None
    try:
        in_bytes = raw
        converted = False
        if is_legacy(lower):
            suffix = lower[lower.rfind("."):] if "." in lower else ".bin"
            fd, tmp_in = tempfile.mkstemp(suffix=suffix)
            os.write(fd, raw)
            os.close(fd)
            out_ext = (
                ".docx"
                if lower.endswith(".doc")
                else ".xlsx"
                if lower.endswith(".xls")
                else ".pptx"
            )
            tmp_dir = os.path.dirname(tmp_in)
            cmd = (
                "libreoffice --headless --convert-to "
                f"{out_ext[1:]} --outdir {shlex.quote(tmp_dir)} "
                f"{shlex.quote(tmp_in)}"
            )
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=False,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode(errors="ignore")[:400]
                raise HTTPException(
                    500, detail=f"LibreOffice convert failed: {err}"
                )
            tmp_out = tmp_in.rsplit(".", 1)[0] + out_ext
            if not os.path.exists(tmp_out):
                raise HTTPException(
                    500, detail="LibreOffice did not produce output file"
                )
            with open(tmp_out, "rb") as f:
                in_bytes = f.read()
            filename = os.path.basename(tmp_out)
            lower = filename.lower()
            converted = True

        # 2) PDF/изображение → стратегии: no-force-OCR vs force-OCR
        if is_pdf(lower) or is_image(lower):
            best_doc = None
            best_tag = None   # 'no_ocr' | 'force_ocr'
            best_score = -1.0

            # (a) Без принудительного OCR; для PDF попробуем OCRmyPDF
            bytes_for_no = in_bytes
            if is_pdf(lower):
                prepped = _try_ocrmypdf_preprocess(in_bytes, _langs)
                if prepped:
                    bytes_for_no = prepped

            conv_no = build_pdf_converter(
                force_ocr=False,
                langs=_langs,
                table_mode=DEFAULT_TABLE_MODE,
            )
            _conv_kwargs = (
                {"max_num_pages": max_pages}
                if isinstance(max_pages, int)
                else {}
            )
            res_no = conv_no.convert(
                _mk_stream(filename, bytes_for_no),
                **_conv_kwargs,
            )
            doc_no = res_no.document
            md_no = (
                doc_no.export_to_markdown()
                if out in ("markdown", "both")
                else ""
            )
            if isinstance(md_no, str):
                md_no, header_removed_no = _detect_and_remove_repeating_header(
                    md_no
                )
                md_no, header_removed_first_no = _remove_first_page_header(
                    md_no
                )
            else:
                header_removed_no = 0
                header_removed_first_no = 0
            text_no = md_no if isinstance(md_no, str) else ""
            total_no = len(text_no)
            if total_no > 0:
                cyr_no = sum(
                    1
                    for ch in text_no
                    if ("А" <= ch <= "я") or ch in "Ёё"
                )
                cyr_ratio_no = cyr_no / total_no
                score_no = total_no * (0.6 + 0.4 * cyr_ratio_no)
            else:
                cyr_ratio_no, score_no = 0.0, 0.0
            trials = [
                {
                    "strategy": "no_ocr",
                    "length": total_no,
                    "cyr_ratio": round(cyr_ratio_no, 3),
                    "score": round(score_no, 3),
                    "header_removed": header_removed_no,
                    "first_header_removed": header_removed_first_no,
                }
            ]
            best_doc = doc_no
            best_tag = "no_ocr"
            best_score = score_no

            # (б) Принудительный полностраничный OCR
            try:
                conv_force = build_pdf_converter(
                    force_ocr=True,
                    langs=_langs,
                    table_mode=DEFAULT_TABLE_MODE,
                )
                res_force = conv_force.convert(
                    _mk_stream("force-ocr-" + filename, in_bytes),
                    **_conv_kwargs,
                )
                doc_force = res_force.document
                md_force = (
                    doc_force.export_to_markdown()
                    if out in ("markdown", "both")
                    else ""
                )
                if isinstance(md_force, str):
                    md_force, header_removed_force = (
                        _detect_and_remove_repeating_header(md_force)
                    )
                    md_force, header_removed_first_force = (
                        _remove_first_page_header(md_force)
                    )
                else:
                    header_removed_force = 0
                    header_removed_first_force = 0
                text_force = md_force if isinstance(md_force, str) else ""
                total_f = len(text_force)
                if total_f > 0:
                    cyr_f = sum(
                        1
                        for ch in text_force
                        if ("А" <= ch <= "я") or ch in "Ёё"
                    )
                    cyr_ratio_f = cyr_f / total_f
                    score_f = total_f * (0.6 + 0.4 * cyr_ratio_f)
                else:
                    cyr_ratio_f, score_f = 0.0, 0.0
                trials.append(
                    {
                        "strategy": "force_ocr",
                        "length": total_f,
                        "cyr_ratio": round(cyr_ratio_f, 3),
                        "score": round(score_f, 3),
                        "header_removed": header_removed_force,
                        "first_header_removed": header_removed_first_force,
                    }
                )
                if score_f > best_score:
                    best_score = score_f
                    best_tag = "force_ocr"
                    best_doc = doc_force
            except Exception as e:
                trials.append({"strategy": "force_ocr", "error": str(e)})

            if best_doc is None:
                raise HTTPException(500, detail="All strategies failed")
            md = (
                best_doc.export_to_markdown()
                if out in ("markdown", "both")
                else None
            )
            js = best_doc.export_to_dict() if out in ("json", "both") else None
            return JSONResponse(content={
                "best_strategy": best_tag,
                "trials": trials,
                "content_markdown": md,
                "content_json": js,
                "meta": {
                    "filename": file.filename,
                    "size_bytes": len(raw),
                    "converted": converted,
                    "out": out,
                    "langs": ",".join(_langs),
                    "max_pages": max_pages,
                }
            })

        # 3) OOXML/HTML/MD/CSV → прямо через Docling,
        #    при ошибке — fallback в PDF
        try:
            conv = build_pdf_converter(
                force_ocr=False,
                langs=_langs,
                table_mode=DEFAULT_TABLE_MODE,
            )
            res = conv.convert(
                _mk_stream(filename, in_bytes),
                **(
                    {"max_num_pages": max_pages}
                    if isinstance(max_pages, int)
                    else {}
                ),
            )
            dl_doc = res.document
            md = (
                dl_doc.export_to_markdown()
                if out in ("markdown", "both")
                else None
            )
            js = dl_doc.export_to_dict() if out in ("json", "both") else None
            return JSONResponse(content={
                "content_markdown": md,
                "content_json": js,
                "meta": {
                    "filename": file.filename,
                    "size_bytes": len(raw),
                    "converted": converted,
                    "out": out,
                    "langs": ",".join(_langs),
                    "max_pages": max_pages,
                }
            })
        except Exception:
            # Пытаемся через LibreOffice → PDF и затем PDF-пайплайн
            try:
                suffix = lower[lower.rfind("."):] if "." in lower else ".bin"
                fd, tmp_in2 = tempfile.mkstemp(suffix=suffix)
                os.write(fd, in_bytes)
                os.close(fd)
                tmp_dir2 = os.path.dirname(tmp_in2)
                cmd2 = (
                    "libreoffice --headless --convert-to pdf --outdir "
                    f"{shlex.quote(tmp_dir2)} {shlex.quote(tmp_in2)}"
                )
                proc2 = subprocess.run(
                    cmd2,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False,
                )
                if proc2.returncode != 0:
                    err2 = proc2.stderr.decode(errors="ignore")[:400]
                    raise HTTPException(
                        500, detail=f"Fallback to PDF failed: {err2}"
                    )
                tmp_pdf = tmp_in2.rsplit(".", 1)[0] + ".pdf"
                if not os.path.exists(tmp_pdf):
                    raise HTTPException(
                        500, detail="Fallback did not produce PDF"
                    )
                with open(tmp_pdf, "rb") as fpdf:
                    pdf_bytes = fpdf.read()
                try:
                    os.remove(tmp_in2)
                    os.remove(tmp_pdf)
                except Exception:
                    pass

                # PDF стратегии для fallback
                conv_no2 = build_pdf_converter(
                    force_ocr=False,
                    langs=_langs,
                    table_mode=DEFAULT_TABLE_MODE,
                )
                _conv_kwargs2 = (
                    {"max_num_pages": max_pages}
                    if isinstance(max_pages, int)
                    else {}
                )
                res_no2 = conv_no2.convert(
                    _mk_stream("fallback.pdf", pdf_bytes),
                    **_conv_kwargs2,
                )
                doc_no2 = res_no2.document
                md_no2 = (
                    doc_no2.export_to_markdown()
                    if out in ("markdown", "both")
                    else ""
                )
                if isinstance(md_no2, str):
                    md_no2, header_removed_no2 = (
                        _detect_and_remove_repeating_header(md_no2)
                    )
                else:
                    header_removed_no2 = 0
                text_no2 = md_no2 if isinstance(md_no2, str) else ""
                total_no2 = len(text_no2)
                if total_no2 > 0:
                    cyr_no2 = sum(
                        1
                        for ch in text_no2
                        if ("А" <= ch <= "я") or ch in "Ёё"
                    )
                    cyr_ratio_no2 = cyr_no2 / total_no2
                    score_no2 = total_no2 * (0.6 + 0.4 * cyr_ratio_no2)
                else:
                    cyr_ratio_no2, score_no2 = 0.0, 0.0

                trials2 = [
                    {
                        "strategy": "no_ocr",
                        "length": total_no2,
                        "cyr_ratio": round(cyr_ratio_no2, 3),
                        "score": round(score_no2, 3),
                        "header_removed": header_removed_no2,
                    }
                ]
                best_doc = doc_no2
                best_tag = "no_ocr"
                best_score = score_no2

                try:
                    conv_force2 = build_pdf_converter(
                        force_ocr=True,
                        langs=_langs,
                        table_mode=DEFAULT_TABLE_MODE,
                    )
                    res_force2 = conv_force2.convert(
                        _mk_stream("fallback-force.pdf", pdf_bytes),
                        **_conv_kwargs2,
                    )
                    doc_force2 = res_force2.document
                    md_force2 = (
                        doc_force2.export_to_markdown()
                        if out in ("markdown", "both")
                        else ""
                    )
                    if isinstance(md_force2, str):
                        md_force2, header_removed_force2 = (
                            _detect_and_remove_repeating_header(md_force2)
                        )
                    else:
                        header_removed_force2 = 0
                    text_force2 = (
                        md_force2 if isinstance(md_force2, str) else ""
                    )
                    total_f2 = len(text_force2)
                    if total_f2 > 0:
                        cyr_f2 = sum(
                            1
                            for ch in text_force2
                            if ("А" <= ch <= "я") or ch in "Ёё"
                        )
                        cyr_ratio_f2 = cyr_f2 / total_f2
                        score_f2 = total_f2 * (0.6 + 0.4 * cyr_ratio_f2)
                    else:
                        cyr_ratio_f2, score_f2 = 0.0, 0.0
                    trials2.append(
                        {
                            "strategy": "force_ocr",
                            "length": total_f2,
                            "cyr_ratio": round(cyr_ratio_f2, 3),
                            "score": round(score_f2, 3),
                            "header_removed": header_removed_force2,
                        }
                    )
                    if score_f2 > best_score:
                        best_doc = doc_force2
                        best_tag = "force_ocr"
                        best_score = score_f2
                except Exception as e:
                    trials2.append({"strategy": "force_ocr", "error": str(e)})

                if best_doc is None:
                    raise HTTPException(
                        500, detail="Fallback strategies failed"
                    )

                md = (
                    best_doc.export_to_markdown()
                    if out in ("markdown", "both")
                    else None
                )
                js = (
                    best_doc.export_to_dict()
                    if out in ("json", "both")
                    else None
                )
                return JSONResponse(content={
                    "best_strategy": best_tag,
                    "trials": trials2,
                    "content_markdown": md,
                    "content_json": js,
                    "meta": {
                        "filename": file.filename,
                        "size_bytes": len(raw),
                        "converted": True,
                        "out": out,
                        "langs": ",".join(_langs),
                        "max_pages": max_pages,
                    }
                })
            except Exception as e:
                raise HTTPException(500, detail=str(e))
    finally:
        try:
            if tmp_in and os.path.exists(tmp_in):
                os.remove(tmp_in)
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
