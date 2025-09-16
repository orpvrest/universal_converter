from app.main import _langs_join, _normalize_easyocr_langs, _normalize_langs


def test_normalize_lang_aliases():
    assert _normalize_langs(["ru", "en"]) == ["rus", "eng"]


def test_normalize_langs_deduplication_and_join():
    langs = _normalize_langs(["ru", "rus", "en", "eng"])
    assert langs == ["rus", "eng"]
    assert _langs_join(langs) == "rus+eng"


def test_normalize_easyocr_langs():
    langs = ["rus", "eng", "ukr", "bel"]
    assert _normalize_easyocr_langs(langs) == ["ru", "en", "uk", "be"]
    # Ensure passthrough for already short codes and deduplication
    assert _normalize_easyocr_langs(["ru", "rus", "en"]) == ["ru", "en"]
