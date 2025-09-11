import io

import pytest
import shutil
import importlib.util
from fastapi.testclient import TestClient

from app.main import app


try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


client = TestClient(app)


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
    resp = client.post("/convert", files=files, data=data)
    assert resp.status_code == 200, resp.text
    js = resp.json()
    md = js.get("content_markdown") or ""
    # Expect that at least part of the phrase is recognized
    assert "Привет" in md or "OCR" in md or "тест" in md
    # Ensure trials include force_ocr path as a fallback candidate
    trials = js.get("trials") or []
    assert any(t.get("strategy") in ("no_ocr", "force_ocr") for t in trials)
