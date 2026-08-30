"""Extraction robuste du texte des fichiers pris en charge (voir formats.py).

L'extraction ne lève jamais d'exception vers l'appelant : un fichier
corrompu, chiffré ou d'un type non pris en charge produit un texte vide plutôt
que d'interrompre le traitement d'un lot entier de fichiers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .formats import SUPPORTED_EXTENSIONS, extract_text_for

# Dossiers de bookkeeping interne à ne jamais reprendre comme documents
# d'entraînement lors d'un scan récursif : `_backup` (doublons mis de côté
# par la détection de doublons, voir `utils.move_files_to_local_backup`) et
# les dossiers d'historique de modèle (`model_store._HISTORY_DIR_NAMES`).
# Sans cette exclusion, un dossier source récursif qui contient (ou dont un
# sous-dossier contient) un `_backup/` réanalyse indéfiniment les mêmes
# fichiers à chaque nouvel entraînement — précisément les fichiers qu'on a
# voulu mettre de côté.
_IGNORED_SCAN_DIR_NAMES = {"_backup", "pkl_history", "json_history", "dataset_history", "__pycache__"}


@dataclass
class ExtractedDocument:
    path: str
    filename: str
    text: str
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def extract_text(path: str) -> tuple[str, str | None]:
    """Retourne (texte, message_erreur). Le texte est vide si l'extraction échoue."""
    try:
        return extract_text_for(path), None
    except Exception as exc:
        return "", str(exc)


def _canonical(path: str) -> str:
    """Chemin absolu, séparateurs normalisés — utilisé comme clé stable pour
    un même fichier physique. Sans cela, un même document réapparaît sous
    deux clés différentes dans le manifest d'un modèle (`model_store.
    model_manifest_path`) selon que le dossier source a été passé en
    relatif/absolu ou avec des `/` plutôt que des `\\`, ce qui empêche
    `discover._sync_dataset` de reconnaître qu'un fichier est déjà suivi :
    il duplique l'entrée et laisse une copie orpheline sous l'ancienne clé,
    jamais nettoyée puisque plus jamais retrouvée."""
    return os.path.normpath(os.path.abspath(path))


def list_documents(
    directory: str, extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS, recursive: bool = False
) -> list[str]:
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Dossier introuvable : {directory}")
    exts = {e.lower() for e in extensions}

    def matches(filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in exts

    if recursive:
        matches_list = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in _IGNORED_SCAN_DIR_NAMES and not d.startswith(".")]
            matches_list.extend(_canonical(os.path.join(root, f)) for f in filenames if matches(f))
        return sorted(matches_list)
    return sorted(_canonical(os.path.join(directory, f)) for f in os.listdir(directory) if matches(f))


def extract_documents_from_paths(paths: list[str]) -> list[ExtractedDocument]:
    """Comme `extract_documents`, mais pour une liste de fichiers précis
    plutôt qu'un dossier entier — utilisé par l'API (`/improve`), où
    l'appelant fournit déjà les chemins exacts."""
    documents = []
    for path in paths:
        canonical_path = _canonical(path)
        text, error = extract_text(canonical_path)
        documents.append(
            ExtractedDocument(path=canonical_path, filename=os.path.basename(canonical_path), text=text, error=error)
        )
    return documents


def extract_documents(
    directory: str,
    recursive: bool = False,
    progress=None,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> list[ExtractedDocument]:
    """`progress`, si fourni, reçoit un message avant le début de
    l'extraction puis à intervalles réguliers pendant qu'elle avance — utile
    pour un gros dossier, où l'extraction seule peut prendre du temps et
    donnerait sinon l'impression que l'outil est figé.

    `extensions` restreint les fichiers pris en compte (voir l'onglet
    Entraînement, sélection des types de fichiers à inclure) : par défaut
    tous les formats pris en charge."""
    paths = list_documents(directory, recursive=recursive, extensions=extensions)
    total = len(paths)
    if progress and total:
        progress(f"Extraction du texte de {total} document(s)...")

    step = max(1, total // 20) if total > 30 else 1
    documents = []
    for i, path in enumerate(paths, start=1):
        text, error = extract_text(path)
        documents.append(
            ExtractedDocument(path=path, filename=os.path.basename(path), text=text, error=error)
        )
        if progress and (i % step == 0 or i == total):
            progress(f"  {i}/{total} document(s) analysé(s)...")
    return documents
