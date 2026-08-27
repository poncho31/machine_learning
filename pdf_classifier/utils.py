"""Écriture atomique et dispatch des fichiers classés.

Un crash ou une interruption (Ctrl+C) en plein milieu d'une écriture directe
laisse un fichier de sortie à moitié écrit et invalide (c'est ce qui est arrivé
à l'ancien classification.json). Écrire dans un fichier temporaire puis le
renommer garantit que le fichier final est soit l'ancienne version, soit la
nouvelle, jamais un état intermédiaire.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile


def write_json_atomic(data, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _unique_destination(path: str) -> str:
    """Si `path` existe déjà, ajoute un suffixe numérique avant l'extension
    (ex. facture.pdf -> facture_1.pdf) jusqu'à trouver un nom libre — pour
    ne jamais écraser silencieusement un fichier déjà présent à destination
    (ex. deux documents source portant le même nom)."""
    if not os.path.exists(path):
        return path
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    suffix = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{suffix}{ext}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def dispatch_file(src_path: str, category: str, output_dir: str, move: bool = False) -> str:
    dest_dir = os.path.join(output_dir, category)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = _unique_destination(os.path.join(dest_dir, os.path.basename(src_path)))
    if move:
        shutil.move(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)
    return dest_path
