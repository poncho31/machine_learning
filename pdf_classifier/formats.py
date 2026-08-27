"""Registre des extracteurs de texte par extension de fichier.

Le reste de l'outil ne connaît pas les types de fichiers spécifiquement : il
délègue l'extraction du texte à l'extracteur enregistré pour l'extension du
fichier. Ajouter un nouveau type de fichier ne demande qu'un nouvel
extracteur ici.

Les formats "texte brut" et ceux qui ne demandent que la bibliothèque
standard (html, xml, rtf, eml...) sont toujours disponibles. Les formats qui
nécessitent un paquet tiers (docx, xlsx/pptx, msg) l'importent en différé et
n'échouent qu'à l'usage, avec un message clair, pour ne jamais imposer une
dépendance lourde à qui ne s'en sert pas.
"""
from __future__ import annotations

import os
import re


def _extract_pdf(path: str) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return " ".join(page.get_text() for page in doc)


def _extract_plain_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _strip_markup(markup_text: str) -> str:
    """Retire les balises HTML/XML d'un texte, ne garde que le contenu."""
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.chunks: list[str] = []

        def handle_data(self, data: str) -> None:
            self.chunks.append(data)

    parser = _TextExtractor()
    parser.feed(markup_text)
    return " ".join(parser.chunks)


def _extract_html(path: str) -> str:
    return _strip_markup(_extract_plain_text(path))


def _extract_xml(path: str) -> str:
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(path)
        return " ".join(t for t in tree.getroot().itertext() if t)
    except ET.ParseError:
        # XML mal formé (ou en fait du HTML) : on retombe sur un nettoyage générique.
        return _strip_markup(_extract_plain_text(path))


_RTF_CONTROL_WORD = re.compile(r"\\[a-zA-Z]+-?\d* ?")
_RTF_HEX_ESCAPE = re.compile(r"\\'[0-9a-fA-F]{2}")
_RTF_BRACES = re.compile(r"[{}]")


def _extract_rtf(path: str) -> str:
    """Suppression grossière des groupes de contrôle RTF. Ne vise pas une
    fidélité parfaite, seulement à récupérer l'essentiel du texte."""
    raw = _extract_plain_text(path)
    text = _RTF_HEX_ESCAPE.sub(" ", raw)
    text = _RTF_CONTROL_WORD.sub(" ", text)
    text = _RTF_BRACES.sub(" ", text)
    return text


def _extract_eml(path: str) -> str:
    from email import policy
    from email.parser import BytesParser

    with open(path, "rb") as f:
        message = BytesParser(policy=policy.default).parse(f)

    parts = [message.get("subject", "") or ""]
    body = message.get_body(preferencelist=("plain", "html"))
    if body is not None:
        content = body.get_content()
        if body.get_content_type() == "text/html":
            content = _strip_markup(content)
        parts.append(content)
    return "\n".join(parts)


def _extract_docx(path: str) -> str:
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError(
            "La lecture des fichiers .docx nécessite le paquet optionnel python-docx : "
            "pip install -r requirements-docx.txt"
        ) from exc
    document = docx.Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_xlsx(path: str) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError(
            "La lecture des fichiers .xlsx nécessite le paquet optionnel openpyxl : "
            "pip install -r requirements-office.txt"
        ) from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    values = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            values.extend(str(cell) for cell in row if cell is not None)
    return " ".join(values)


def _extract_pptx(path: str) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "La lecture des fichiers .pptx nécessite le paquet optionnel python-pptx : "
            "pip install -r requirements-office.txt"
        ) from exc
    presentation = Presentation(path)
    parts = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _extract_msg(path: str) -> str:
    try:
        import extract_msg
    except ImportError as exc:
        raise RuntimeError(
            "La lecture des fichiers .msg (Outlook) nécessite le paquet optionnel extract-msg : "
            "pip install -r requirements-msg.txt"
        ) from exc
    message = extract_msg.Message(path)
    return "\n".join(part for part in (message.subject, message.body) if part)


# Extension (minuscule, avec le point) -> fonction d'extraction(path) -> texte.
EXTRACTORS = {
    # Documents
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".rtf": _extract_rtf,
    # Texte brut et données structurées lisibles telles quelles
    ".txt": _extract_plain_text,
    ".md": _extract_plain_text,
    ".csv": _extract_plain_text,
    ".tsv": _extract_plain_text,
    ".log": _extract_plain_text,
    ".json": _extract_plain_text,
    ".yaml": _extract_plain_text,
    ".yml": _extract_plain_text,
    ".ini": _extract_plain_text,
    ".cfg": _extract_plain_text,
    ".toml": _extract_plain_text,
    # Balisage
    ".html": _extract_html,
    ".htm": _extract_html,
    ".xml": _extract_xml,
    # Email
    ".eml": _extract_eml,
    ".msg": _extract_msg,
    # Bureautique (tableurs, présentations)
    ".xlsx": _extract_xlsx,
    ".xlsm": _extract_xlsx,
    ".pptx": _extract_pptx,
}

SUPPORTED_EXTENSIONS = tuple(sorted(EXTRACTORS))


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in EXTRACTORS


def extract_text_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Type de fichier non pris en charge : {ext or path}")
    return extractor(path)
