"""Docling FastAPI microservice.

This service converts various office documents, PDFs, and images into
structured representations (Markdown and/or JSON) using Docling.
It includes format autodetection, optional legacy conversion via LibreOffice
(.doc/.xls/.ppt to OOXML), and an OCR strategy sweep for image-like inputs
using Tesseract.

Environment variables:
    UVICORN_HOST (str): Host for uvicorn. Default: 0.0.0.0
    UVICORN_PORT (int): Port for uvicorn. Default: 8088
    OMP_NUM_THREADS (int): OpenMP threads for OCR/CPU-bound ops. Default: 4
    DOCLING_ARTIFACTS_PATH (str): Directory for Docling models.
    DOCLING_TABLE_MODE (str): Table extraction mode, e.g., "ACCURATE".
    DOCLING_FORCE_OCR (bool): Whether to force OCR (baseline default).
        Default: true
    DOCLING_LANGS (str): Comma-separated OCR languages, e.g., "rus,eng".
    MAX_FILE_SIZE_MB (int): Max upload size in MB. Default: 80

Main endpoints:
    GET /health
        Liveness check.

    POST /convert
        Universal endpoint: autodetect strategy per file-type.
        - Legacy .doc/.xls/.ppt → LibreOffice → OOXML → Docling (no OCR)
        - PDF/images → try no-OCR then force-OCR with PSM candidates; pick best
        - OOXML/HTML/MD/CSV → Docling (no OCR)

"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import tempfile
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TableFormerMode,
)

app = FastAPI(title="Docling microservice", version="3.0.0")

DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH", None)
DEFAULT_TABLE_MODE = os.getenv("DOCLING_TABLE_MODE", "ACCURATE").upper()
DEFAULT_FORCE_OCR = os.getenv("DOCLING_FORCE_OCR", "true").lower() == "true"
DEFAULT_LANGS = [
    x.strip()
    for x in os.getenv("DOCLING_LANGS", "rus,eng").split(",")
    if x.strip()
]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "80"))


# Helper to construct a DocumentConverter for PDF/image scenarios
def build_pdf_converter(
    force_ocr: bool,
    langs: list[str],
    table_mode: str,
    psm: Optional[int] = None,
) -> DocumentConverter:
    """Build a DocumentConverter tailored to PDF/image inputs.

    Args:
        force_ocr: Whether to force full-page OCR.
        langs: OCR languages list (e.g., ["rus", "eng"]).
        table_mode: Table structure mode (e.g., "ACCURATE").
        psm: Optional page segmentation mode for Tesseract.

    Returns:
        DocumentConverter configured for PDF/image processing.
    """
    pipe = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS_PATH)

    # When not forcing OCR, let Docling decide; still pass langs but don't
    # force full-page
    if force_ocr:
        pipe.do_ocr = True
        pipe.force_full_page_ocr = True
        ocr_opts = TesseractCliOcrOptions(
            lang=langs,
            force_full_page_ocr=True,
        )
        if psm is not None:
            if hasattr(ocr_opts, "page_seg_mode"):
                ocr_opts.page_seg_mode = int(psm)
            else:
                ocr_opts.extra_args = (ocr_opts.extra_args or []) + [
                    "--psm",
                    str(psm),
                ]
        pipe.ocr_options = ocr_opts
    else:
        pipe.do_ocr = True
        pipe.force_full_page_ocr = False
        ocr_opts = TesseractCliOcrOptions(
            lang=langs,
            force_full_page_ocr=False,
        )
        pipe.ocr_options = ocr_opts

    if table_mode == "ACCURATE":
        pipe.do_table_structure = True
        pipe.table_structure_options.mode = TableFormerMode.ACCURATE
        pipe.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)
        }
    )


class ConvertResponse(BaseModel):
    """Standard conversion response model.

    Attributes:
        content_markdown: Output Markdown string (if requested).
        content_json: Output JSON structure (if requested).
        meta: Metadata about the request/processing.
    """
    content_markdown: Optional[str] = None
    content_json: Optional[dict] = None
    meta: dict


@app.get("/health")
def health():
    """Health check endpoint.

    Returns:
        dict: {"ok": True}
    """
    return {"ok": True}

# Universal endpoint: autodetect + legacy convert + OCR PSM sweep


@app.post("/convert", response_model=ConvertResponse)
async def convert_universal(
    file: UploadFile = File(...),
    out: Literal["markdown", "json", "both"] = Form("both"),
    langs: Optional[str] = Form(None),              # "rus,eng" | "auto"
    psm_list: str = Form("6,4,11"),               # PSM для OCR-форматов
    max_pages: Optional[int] = Form(None),
):
    """Универсальная конвертация документов с авто-стратегией.

    Args:
        file: Загружаемый файл (любой офисный формат, PDF или изображение).
        out: Формат вывода: "markdown", "json" или "both".
        langs: Языки OCR (строка, например "rus,eng"). Если None — берём из
            переменной окружения DOCLING_LANGS.
        psm_list: Список PSM для Tesseract, через запятую (например, "6,4,11").
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

        trials = []

        # 2) PDF/изображение → стратегии: no-force-OCR vs force-OCR (PSM sweep)
        if is_pdf(lower) or is_image(lower):
            best_doc = None
            best_tag = None   # 'no_ocr' | 'psm:<n>'
            best_score = -1.0

            # (a) Без принудительного OCR (Docling сам решит)
            conv_no = build_pdf_converter(
                force_ocr=False,
                langs=_langs,
                table_mode=DEFAULT_TABLE_MODE,
            )
            res_no = conv_no.convert(
                DocumentStream(name=filename, stream=io.BytesIO(in_bytes)),
                max_num_pages=max_pages,
            )
            doc_no = res_no.document
            md_no = (
                doc_no.export_to_markdown()
                if out in ("markdown", "both")
                else ""
            )
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
            trials.append(
                {
                    "strategy": "no_ocr",
                    "length": total_no,
                    "cyr_ratio": round(cyr_ratio_no, 3),
                    "score": round(score_no, 3),
                }
            )
            if score_no > best_score:
                best_score, best_tag, best_doc = score_no, 'no_ocr', doc_no

            # (б) Force OCR + PSM candidates
            candidates = [x.strip() for x in psm_list.split(",") if x.strip()]
            try:
                psm_candidates = [int(x) for x in candidates]
            except Exception:
                psm_candidates = [6, 4, 11]
            if not psm_candidates:
                psm_candidates = [6, 4, 11]

            for psm in psm_candidates:
                try:
                    conv_psm = build_pdf_converter(
                        force_ocr=True,
                        langs=_langs,
                        table_mode=DEFAULT_TABLE_MODE,
                        psm=psm,
                    )
                    res_psm = conv_psm.convert(
                        DocumentStream(
                            name=f"psm{psm}-" + filename,
                            stream=io.BytesIO(in_bytes),
                        ),
                        max_num_pages=max_pages,
                    )
                    doc_psm = res_psm.document
                    md_psm = (
                        doc_psm.export_to_markdown()
                        if out in ("markdown", "both")
                        else ""
                    )
                    text_psm = md_psm if isinstance(md_psm, str) else ""
                    total = len(text_psm)
                    if total > 0:
                        cyr = sum(
                            1
                            for ch in text_psm
                            if ("А" <= ch <= "я") or ch in "Ёё"
                        )
                        cyr_ratio = cyr / total
                        score = total * (0.6 + 0.4 * cyr_ratio)
                    else:
                        cyr_ratio, score = 0.0, 0.0
                    trials.append(
                        {
                            "strategy": f"psm:{psm}",
                            "length": total,
                            "cyr_ratio": round(cyr_ratio, 3),
                            "score": round(score, 3),
                        }
                    )
                    if score > best_score:
                        best_score, best_tag, best_doc = (
                            score,
                            f"psm:{psm}",
                            doc_psm,
                        )
                except Exception as e:
                    trials.append(
                        {
                            "strategy": f"psm:{psm}",
                            "error": str(e),
                        }
                    )

            if best_doc is None:
                raise HTTPException(500, detail="All strategies failed")
            md = (
                best_doc.export_to_markdown()
                if out in ("markdown", "both")
                else None
            )
            js = (
                best_doc.export_to_dict() if out in ("json", "both") else None
            )
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

        # 3) OOXML/HTML/MD/CSV → прямо через Docling
        conv = build_pdf_converter(
            force_ocr=False,
            langs=_langs,
            table_mode=DEFAULT_TABLE_MODE,
        )
        res = conv.convert(
            DocumentStream(name=filename, stream=io.BytesIO(in_bytes)),
            max_num_pages=max_pages,
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
    finally:
        try:
            if tmp_in and os.path.exists(tmp_in):
                os.remove(tmp_in)
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
