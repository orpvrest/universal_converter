"""Утилиты подготовки моделей Docling и вспомогательные функции OCR."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .config import DOCLING_ARTIFACTS_PATH, DEFAULT_TABLE_MODE
from .langs import langs_join, normalize_easyocr_langs

_MODELS_READY = False


def ensure_models() -> None:
    """Подготавливает артефакты Docling и гарантирует наличие моделей."""
    global _MODELS_READY
    if _MODELS_READY or not DOCLING_ARTIFACTS_PATH:
        _MODELS_READY = True if DOCLING_ARTIFACTS_PATH else _MODELS_READY
        return

    try:
        import importlib

        mdl = importlib.import_module("docling.utils.model_downloader")
        download_models = getattr(mdl, "download_models")

        root = Path(DOCLING_ARTIFACTS_PATH)
        root.mkdir(parents=True, exist_ok=True)

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

        root_st = root / "model.safetensors"
        if not root_st.exists() and found_any:
            target = found_any[0]
            try:
                root_st.symlink_to(target)
            except Exception:
                try:
                    shutil.copyfile(target, root_st)
                except Exception:
                    pass

        allowed_types = {
            "conditional_detr",
            "dfine",
            "dab_detr",
            "deformable_detr",
            "deta",
            "detr",
            "rtdetr",
            "rtdetrv2",
            "table-transformer",
            "yolos",
        }

        def _is_layout_dir(path: Path) -> tuple[bool, str | None]:
            cfg = path / "config.json"
            if not cfg.exists():
                return False, None
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                mt = (data.get("model_type") or "").lower()
                return (mt in allowed_types), mt
            except Exception:
                return False, None

        candidates_dirs: list[Path] = []
        for st in found_any:
            candidates_dirs.append(st.parent)
        try:
            for cfg_path in root.rglob("config.json"):
                candidates_dirs.append(cfg_path.parent)
        except Exception:
            pass
        seen = set()
        uniq_dirs: list[Path] = []
        for d in candidates_dirs:
            if str(d) not in seen:
                seen.add(str(d))
                uniq_dirs.append(d)

        model_dir: Path | None = None
        for cand in uniq_dirs:
            ok, _ = _is_layout_dir(cand)
            if ok:
                model_dir = cand
                break
        if model_dir is None and found_any:
            model_dir = found_any[0].parent

        if model_dir is not None:
            needed = (
                "preprocessor_config.json",
                "config.json",
                "model.safetensors",
            )
            for name in needed:
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

        pc_root = root / "preprocessor_config.json"
        if not pc_root.exists():
            try:
                pc_any = next(root.rglob("preprocessor_config.json"), None)
            except Exception:
                pc_any = None
            if pc_any is not None and pc_any != pc_root:
                try:
                    pc_root.symlink_to(pc_any)
                except Exception:
                    try:
                        shutil.copyfile(pc_any, pc_root)
                    except Exception:
                        pass

        if not (root / "preprocessor_config.json").exists():
            try:
                download_models(
                    output_dir=root,
                    progress=False,
                    with_layout=True,
                    with_tableformer=True,
                    with_code_formula=True,
                    with_picture_classifier=True,
                    with_easyocr=True,
                )
            except Exception:
                pass

        try:
            print(
                "ensure_models: artifacts_root=",
                str(root),
                " root_has_preprocessor=",
                (root / "preprocessor_config.json").exists(),
                " root_has_model=",
                (root / "model.safetensors").exists(),
                " selected_dir=",
                str(model_dir) if model_dir else None,
            )
        except Exception:
            pass

        _MODELS_READY = True
    except Exception:
        _MODELS_READY = True


def try_ocrmypdf_preprocess(pdf_bytes: bytes, langs: list[str]) -> Optional[bytes]:
    """Попытка прогнать PDF через OCRmyPDF для выравнивания и очистки."""
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
            "--clean-final",
            "--threshold",
            "--remove-background",
            "--optimize",
            "3",
            "--language",
            langs_join(langs),
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
        data = Path(out_path).read_bytes()
        return data if len(data) > 1024 else None
    except Exception:
        return None
    finally:
        for path in (in_path, out_path):
            try:
                os.remove(path)
            except Exception:
                pass


def build_pdf_converter(
    force_ocr: bool,
    langs: list[str],
    table_mode: str = DEFAULT_TABLE_MODE,
    ocr_engine: str = "tesseract",
    force_table_mode: str | None = None,
) -> Any:
    """Собирает DocumentConverter для PDF/изображений."""
    ensure_models()
    import importlib

    dc_mod = importlib.import_module("docling.document_converter")
    dm_base = importlib.import_module("docling.datamodel.base_models")
    pipe_mod = importlib.import_module("docling.datamodel.pipeline_options")
    DocumentConverter = getattr(dc_mod, "DocumentConverter")
    PdfFormatOption = getattr(dc_mod, "PdfFormatOption")
    InputFormat = getattr(dm_base, "InputFormat")
    PdfPipelineOptions = getattr(pipe_mod, "PdfPipelineOptions")
    TesseractCliOcrOptions = getattr(pipe_mod, "TesseractCliOcrOptions")
    EasyOcrOptions = getattr(pipe_mod, "EasyOcrOptions")
    TableFormerMode = getattr(pipe_mod, "TableFormerMode")

    pipe = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS_PATH)
    if force_ocr:
        pipe.do_ocr = True
        if ocr_engine == "easyocr":
            easy_langs = normalize_easyocr_langs(langs)
            ocr_opts = EasyOcrOptions(
                lang=easy_langs,
                force_full_page_ocr=True,
            )
        else:
            ocr_opts = TesseractCliOcrOptions(
                lang=langs,
                force_full_page_ocr=True,
            )
        pipe.ocr_options = ocr_opts
    else:
        pipe.do_ocr = False

    effective_table_mode = (
        force_table_mode if force_ocr and force_table_mode else table_mode
    )
    effective_table_mode = (effective_table_mode or "").upper()

    if effective_table_mode:
        try:
            table_mode_enum = getattr(TableFormerMode, effective_table_mode)
        except AttributeError:
            table_mode_enum = getattr(TableFormerMode, "ACCURATE")
            effective_table_mode = "ACCURATE"
        pipe.do_table_structure = True
        pipe.table_structure_options.mode = table_mode_enum
        if effective_table_mode not in {"LITE", "NONE"}:
            pipe.table_structure_options.do_cell_matching = True
        if effective_table_mode == "NONE":
            pipe.do_table_structure = False
    elif table_mode == "ACCURATE":  # backward compatibility default
        pipe.do_table_structure = True
        pipe.table_structure_options.mode = TableFormerMode.ACCURATE
        pipe.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipe),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipe),
        }
    )


# Обратная совместимость с прежними именами.
_try_ocrmypdf_preprocess = try_ocrmypdf_preprocess
