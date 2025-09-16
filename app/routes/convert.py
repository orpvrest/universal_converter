"""Маршруты универсальной конвертации документов."""

from __future__ import annotations

import io
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import DEFAULT_TABLE_MODE, FORCE_OCR_TABLE_MODE, MAX_FILE_SIZE_MB
from ..langs import resolve_langs_param
from ..markdown_filters import (
    detect_and_remove_repeating_header,
    remove_first_page_header,
)
from ..ocr_utils import build_pdf_converter, try_ocrmypdf_preprocess
from ..schemas import ConvertResponse

router = APIRouter()


@router.post("/convert", response_model=ConvertResponse)
async def convert_universal(
    file: UploadFile = File(...),
    out: Literal["markdown", "json", "both"] = Form("both"),
    langs: Optional[str] = Form(None),
    max_pages: Optional[int] = Form(None),
):
    """Универсальная конвертация документов с авто-стратегией."""
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
    langs_list = resolve_langs_param(langs)

    def _mk_stream(name: str, data: bytes):
        import importlib

        dm_base = importlib.import_module("docling.datamodel.base_models")
        DocumentStream = getattr(dm_base, "DocumentStream")
        return DocumentStream(name=name, stream=io.BytesIO(data))

    underscore_re = re.compile(r"[\s]*[_\-]{3,}[\s]*")
    multi_space_re = re.compile(r"\s{2,}")
    artifact_line_re = re.compile(r"^[ \t]*[_\-]{3,}[ \t]*$", re.MULTILINE)
    artifact_inline_re = re.compile(r"[_\-]{3,}")

    def _sanitize_for_scoring(text: str) -> str:
        cleaned = underscore_re.sub(" ", text)
        cleaned = multi_space_re.sub(" ", cleaned)
        return cleaned.strip()

    def _cleanup_markdown_artifacts(md_text: str) -> str:
        without_lines = artifact_line_re.sub("", md_text)
        return artifact_inline_re.sub(" ", without_lines)

    def _build_trial(
        tag: str,
        md_text: str | None,
        header_removed: int = 0,
        header_removed_first: int = 0,
    ) -> tuple[dict[str, object], float]:
        if not isinstance(md_text, str) or not md_text:
            trial = {
                "strategy": tag,
                "length": 0,
                "cyr_ratio": 0.0,
                "digit_ratio": 0.0,
                "score": 0.0,
                "header_removed": header_removed,
                "first_header_removed": header_removed_first,
            }
            return trial, 0.0

        sanitized = _sanitize_for_scoring(md_text)
        total = len(sanitized)
        if total == 0:
            trial = {
                "strategy": tag,
                "length": 0,
                "cyr_ratio": 0.0,
                "digit_ratio": 0.0,
                "score": 0.0,
                "header_removed": header_removed,
                "first_header_removed": header_removed_first,
            }
            return trial, 0.0

        base = sanitized.lower()
        cyr = sum(1 for ch in base if ("а" <= ch <= "я") or ch == "ё")
        digits = sum(1 for ch in base if ch.isdigit())
        cyr_ratio = cyr / total
        digit_ratio = digits / total
        score = total * (0.5 + 0.35 * cyr_ratio + 0.15 * digit_ratio)
        trial = {
            "strategy": tag,
            "length": total,
            "cyr_ratio": round(cyr_ratio, 3),
            "digit_ratio": round(digit_ratio, 3),
            "score": round(score, 3),
            "header_removed": header_removed,
            "first_header_removed": header_removed_first,
        }
        return trial, score

    def is_legacy(name: str) -> bool:
        return name.endswith((".doc", ".xls", ".ppt"))

    def is_pdf(name: str) -> bool:
        return name.endswith(".pdf")

    def is_image(name: str) -> bool:
        return name.endswith((
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
        ))

    def is_direct_supported(name: str) -> bool:
        return name.endswith(
            (
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

    tmp_in = tmp_out = None
    try:
        in_bytes = raw
        converted = False
        if is_legacy(lower):
            suffix = lower[lower.rfind(".") :] if "." in lower else ".bin"
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

        if is_pdf(lower) or is_image(lower):
            best_doc = None
            best_tag = None
            best_score = -1.0

            bytes_for_no = in_bytes
            if is_pdf(lower):
                prepped = try_ocrmypdf_preprocess(in_bytes, langs_list)
                if prepped:
                    bytes_for_no = prepped

            conv_no = build_pdf_converter(
                force_ocr=False,
                langs=langs_list,
                table_mode=DEFAULT_TABLE_MODE,
            )
            conv_kwargs = (
                {"max_num_pages": max_pages}
                if isinstance(max_pages, int)
                else {}
            )
            res_no = conv_no.convert(
                _mk_stream(filename, bytes_for_no),
                **conv_kwargs,
            )
            doc_no = res_no.document
            md_no = (
                doc_no.export_to_markdown()
                if out in ("markdown", "both")
                else ""
            )
            if isinstance(md_no, str):
                md_no, header_removed_no = detect_and_remove_repeating_header(md_no)
                md_no, header_removed_first_no = remove_first_page_header(md_no)
                md_no = _cleanup_markdown_artifacts(md_no)
            else:
                header_removed_no = 0
                header_removed_first_no = 0
            trial_no, score_no = _build_trial(
                "no_ocr", md_no, header_removed_no, header_removed_first_no
            )
            trials = [trial_no]
            best_doc = doc_no
            best_tag = "no_ocr"
            best_score = score_no

            try:
                conv_force = build_pdf_converter(
                    force_ocr=True,
                    langs=langs_list,
                    table_mode=DEFAULT_TABLE_MODE,
                    force_table_mode=FORCE_OCR_TABLE_MODE,
                )
                res_force = conv_force.convert(
                    _mk_stream("force-ocr-" + filename, in_bytes),
                    **conv_kwargs,
                )
                doc_force = res_force.document
                md_force = (
                    doc_force.export_to_markdown()
                    if out in ("markdown", "both")
                    else ""
                )
                if isinstance(md_force, str):
                    md_force, header_removed_force = detect_and_remove_repeating_header(
                        md_force
                    )
                    md_force, header_removed_first_force = remove_first_page_header(
                        md_force
                    )
                    md_force = _cleanup_markdown_artifacts(md_force)
                else:
                    header_removed_force = 0
                    header_removed_first_force = 0
                trial_force, score_f = _build_trial(
                    "force_ocr",
                    md_force,
                    header_removed_force,
                    header_removed_first_force,
                )
                trials.append(trial_force)
                if score_f > best_score:
                    best_score = score_f
                    best_tag = "force_ocr"
                    best_doc = doc_force
            except Exception as exc:
                trials.append({"strategy": "force_ocr", "error": str(exc)})

            try:
                conv_easy = build_pdf_converter(
                    force_ocr=True,
                    langs=langs_list,
                    table_mode=DEFAULT_TABLE_MODE,
                    force_table_mode=FORCE_OCR_TABLE_MODE,
                    ocr_engine="easyocr",
                )
                res_easy = conv_easy.convert(
                    _mk_stream("easyocr-" + filename, in_bytes),
                    **conv_kwargs,
                )
                doc_easy = res_easy.document
                md_easy = (
                    doc_easy.export_to_markdown()
                    if out in ("markdown", "both")
                    else ""
                )
                if isinstance(md_easy, str):
                    md_easy, header_removed_easy = detect_and_remove_repeating_header(
                        md_easy
                    )
                    md_easy, header_removed_first_easy = remove_first_page_header(
                        md_easy
                    )
                    md_easy = _cleanup_markdown_artifacts(md_easy)
                else:
                    header_removed_easy = 0
                    header_removed_first_easy = 0
                trial_easy, score_e = _build_trial(
                    "easyocr",
                    md_easy,
                    header_removed_easy,
                    header_removed_first_easy,
                )
                trials.append(trial_easy)
                if score_e > best_score:
                    best_score = score_e
                    best_tag = "easyocr"
                    best_doc = doc_easy
            except Exception as exc:
                trials.append({"strategy": "easyocr", "error": str(exc)})

            if best_doc is None:
                raise HTTPException(500, detail="All strategies failed")
            md = (
                best_doc.export_to_markdown()
                if out in ("markdown", "both")
                else None
            )
            if isinstance(md, str):
                md = _cleanup_markdown_artifacts(md)
            js = best_doc.export_to_dict() if out in ("json", "both") else None
            return JSONResponse(
                content={
                    "best_strategy": best_tag,
                    "trials": trials,
                    "content_markdown": md,
                    "content_json": js,
                    "meta": {
                        "filename": file.filename,
                        "size_bytes": len(raw),
                        "converted": converted,
                        "out": out,
                        "langs": ",".join(langs_list),
                        "max_pages": max_pages,
                    },
                }
            )

        try:
            conv = build_pdf_converter(
                force_ocr=False,
                langs=langs_list,
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
            if isinstance(md, str):
                md = _cleanup_markdown_artifacts(md)
            js = dl_doc.export_to_dict() if out in ("json", "both") else None
            return JSONResponse(
                content={
                    "content_markdown": md,
                    "content_json": js,
                    "meta": {
                        "filename": file.filename,
                        "size_bytes": len(raw),
                        "converted": converted,
                        "out": out,
                        "langs": ",".join(langs_list),
                        "max_pages": max_pages,
                    },
                }
            )
        except Exception:
            try:
                suffix = lower[lower.rfind(".") :] if "." in lower else ".bin"
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

                conv_no2 = build_pdf_converter(
                    force_ocr=False,
                    langs=langs_list,
                    table_mode=DEFAULT_TABLE_MODE,
                )
                conv_kwargs2 = (
                    {"max_num_pages": max_pages}
                    if isinstance(max_pages, int)
                    else {}
                )
                res_no2 = conv_no2.convert(
                    _mk_stream("fallback.pdf", pdf_bytes),
                    **conv_kwargs2,
                )
                doc_no2 = res_no2.document
                md_no2 = (
                    doc_no2.export_to_markdown()
                    if out in ("markdown", "both")
                    else ""
                )
                if isinstance(md_no2, str):
                    md_no2, header_removed_no2 = detect_and_remove_repeating_header(
                        md_no2
                    )
                    md_no2 = _cleanup_markdown_artifacts(md_no2)
                else:
                    header_removed_no2 = 0

                trial_no2, score_no2 = _build_trial(
                    "no_ocr", md_no2, header_removed_no2, 0
                )

                trials2 = [trial_no2]
                best_doc = doc_no2
                best_tag = "no_ocr"
                best_score = score_no2

                try:
                    conv_force2 = build_pdf_converter(
                        force_ocr=True,
                        langs=langs_list,
                        table_mode=DEFAULT_TABLE_MODE,
                        force_table_mode=FORCE_OCR_TABLE_MODE,
                    )
                    res_force2 = conv_force2.convert(
                        _mk_stream("fallback-force.pdf", pdf_bytes),
                        **conv_kwargs2,
                    )
                    doc_force2 = res_force2.document
                    md_force2 = (
                        doc_force2.export_to_markdown()
                        if out in ("markdown", "both")
                        else ""
                    )
                    if isinstance(md_force2, str):
                        md_force2, header_removed_force2 = (
                            detect_and_remove_repeating_header(md_force2)
                        )
                        md_force2 = _cleanup_markdown_artifacts(md_force2)
                    else:
                        header_removed_force2 = 0
                    trial_force2, score_f2 = _build_trial(
                        "force_ocr", md_force2, header_removed_force2, 0
                    )
                    trials2.append(trial_force2)
                    if score_f2 > best_score:
                        best_doc = doc_force2
                        best_tag = "force_ocr"
                        best_score = score_f2
                except Exception as exc:
                    trials2.append({"strategy": "force_ocr", "error": str(exc)})

                try:
                    conv_easy2 = build_pdf_converter(
                        force_ocr=True,
                        langs=langs_list,
                        table_mode=DEFAULT_TABLE_MODE,
                        force_table_mode=FORCE_OCR_TABLE_MODE,
                        ocr_engine="easyocr",
                    )
                    res_easy2 = conv_easy2.convert(
                        _mk_stream("fallback-easy.pdf", pdf_bytes),
                        **conv_kwargs2,
                    )
                    doc_easy2 = res_easy2.document
                    md_easy2 = (
                        doc_easy2.export_to_markdown()
                        if out in ("markdown", "both")
                        else ""
                    )
                    if isinstance(md_easy2, str):
                        md_easy2, header_removed_easy2 = (
                            detect_and_remove_repeating_header(md_easy2)
                        )
                        md_easy2 = _cleanup_markdown_artifacts(md_easy2)
                    else:
                        header_removed_easy2 = 0
                    trial_easy2, score_e2 = _build_trial(
                        "easyocr", md_easy2, header_removed_easy2, 0
                    )
                    trials2.append(trial_easy2)
                    if score_e2 > best_score:
                        best_doc = doc_easy2
                        best_tag = "easyocr"
                        best_score = score_e2
                except Exception as exc:
                    trials2.append({"strategy": "easyocr", "error": str(exc)})

                if best_doc is None:
                    raise HTTPException(
                        500, detail="Fallback strategies failed"
                    )

                md = (
                    best_doc.export_to_markdown()
                    if out in ("markdown", "both")
                    else None
                )
                if isinstance(md, str):
                    md = _cleanup_markdown_artifacts(md)
                js = (
                    best_doc.export_to_dict()
                    if out in ("json", "both")
                    else None
                )
                return JSONResponse(
                    content={
                        "best_strategy": best_tag,
                        "trials": trials2,
                        "content_markdown": md,
                        "content_json": js,
                        "meta": {
                            "filename": file.filename,
                            "size_bytes": len(raw),
                            "converted": True,
                            "out": out,
                            "langs": ",".join(langs_list),
                            "max_pages": max_pages,
                        },
                    }
                )
            except Exception as exc:
                raise HTTPException(500, detail=str(exc))
    finally:
        try:
            if tmp_in and os.path.exists(tmp_in):
                os.remove(tmp_in)
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
