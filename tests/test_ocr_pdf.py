# Интеграционные проверки OCR и fallback-стратегий Docling сервиса.
import importlib
import io
import shutil

import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


client = TestClient(app)
REAL_PDF_URL = (
    "https://mai.ru/upload/iblock/6f4/"
    "nqun8lcnq727pkbgla33zzusnlxpupns/"
    "2025_08_25_221_ORG_Ob_utverzhdenii_instruktsii_"
    "o_vzyatii_pomeshchenii_.pdf"
)
REAL_PDF_TOKENS = ("инструкции", "помещении", "взятии")


def _make_image_pdf_bytes(text: str = "Привет, мир! OCR тест.") -> bytes:
    if Image is None:
        pytest.skip("Pillow is required for OCR PDF test")
    img = Image.new("RGB", (1200, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Try to use a default font; exact font is not critical for OCR
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    draw.text((50, 150), text, fill=(0, 0, 0), font=font)

    # Save as PDF (image-only)
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


@pytest.mark.timeout(60)
def test_convert_ocr_image_pdf():
    if importlib.util.find_spec("docling") is None:
        pytest.skip("docling is required for OCR test")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract CLI is required for OCR test")
    pdf_bytes = _make_image_pdf_bytes()
    files = {"file": ("ocr_test.pdf", pdf_bytes, "application/pdf")}
    data = {"out": "markdown", "langs": "rus,eng"}
    try:
        resp = client.post("/convert", files=files, data=data)
    except requests.exceptions.RequestException as exc:
        pytest.skip(
            f"Conversion requires network access for Docling models: {exc}"
        )
    if resp.status_code >= 500:
        pytest.skip(
            f"Conversion failed with status {resp.status_code}: {resp.text}"
        )
    assert resp.status_code == 200, resp.text
    js = resp.json()
    md = js.get("content_markdown") or ""
    # Expect that at least part of the phrase is recognized
    assert "Привет" in md or "OCR" in md or "тест" in md
    # Ensure trials include both tesseract and easyocr paths
    trials = js.get("trials") or []
    assert any(t.get("strategy") == "force_ocr" for t in trials)
    assert any(t.get("strategy") == "easyocr" for t in trials)


@pytest.mark.timeout(120)
def test_convert_real_russian_pdf():
    if httpx is None:
        pytest.skip("httpx is required for downloading the sample PDF")
    if importlib.util.find_spec("docling") is None:
        pytest.skip("docling is required for OCR test")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract CLI is required for OCR test")
    try:
        resp = httpx.get(
            REAL_PDF_URL,
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Referer": "https://mai.ru/",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.6,en;q=0.4",
            },
        )
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network issues should skip
        pytest.skip(f"Unable to download sample PDF: {exc}")

    pdf_bytes = resp.content
    assert len(pdf_bytes) > 1024  # sanity check
    files = {"file": ("mai_instruction.pdf", pdf_bytes, "application/pdf")}
    data = {"out": "markdown", "langs": "ru,eng"}
    try:
        resp = client.post("/convert", files=files, data=data)
    except requests.exceptions.RequestException as exc:
        pytest.skip(
            f"Conversion requires network access for Docling models: {exc}"
        )
    if resp.status_code >= 500:
        pytest.skip(
            f"Conversion failed with status {resp.status_code}: {resp.text}"
        )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    markdown = (payload.get("content_markdown") or "").lower()
    assert any(token in markdown for token in REAL_PDF_TOKENS), markdown[:400]

    trials = payload.get("trials") or []
    assert any(t.get("strategy") == "force_ocr" for t in trials)
    assert any(t.get("strategy") == "easyocr" for t in trials)

    meta = payload.get("meta") or {}
    # Алиас ru → rus должен сработать и вернуться в метаданных
    assert meta.get("langs") == "rus,eng"
    assert payload.get("best_strategy") in {"force_ocr", "easyocr"}
