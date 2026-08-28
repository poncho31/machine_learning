"""Écriture atomique et dispatch des fichiers classés.

Un crash ou une interruption (Ctrl+C) en plein milieu d'une écriture directe
laisse un fichier de sortie à moitié écrit et invalide (c'est ce qui est arrivé
à l'ancien classification.json). Écrire dans un fichier temporaire puis le
renommer garantit que le fichier final est soit l'ancienne version, soit la
nouvelle, jamais un état intermédiaire.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile


def content_hash(text: str) -> str:
    """Empreinte du contenu d'un document, indépendante de son chemin.

    Une correction confirmée à la main doit rester attachée au *contenu* du
    document, pas à son chemin : celui-ci change à chaque export (dossier de
    classification horodaté, dossier `dataset/` du modèle...), alors que le
    texte extrait d'un même fichier reste identique d'un passage à l'autre —
    c'est ce qui permet à `classify()` de reconnaître un document déjà
    corrigé lors d'une classification ultérieure du même dossier."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


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


def detect_duplicate_pairs(
    paths: list[str],
    filenames: list[str],
    vectors,
    threshold: float = 0.97,
    max_docs: int = 4000,
) -> list[dict]:
    """Paires de documents quasi identiques par similarité cosinus, à partir
    de vecteurs déjà L2-normalisés (TF-IDF comme embeddings) : la similarité
    cosinus s'y réduit à un simple produit scalaire, donc gratuite à partir
    de vecteurs déjà calculés ailleurs (regroupement à l'entraînement,
    prédiction en classification). Réutilisé par l'onglet Entraînement et
    l'onglet Classification. Coût en O(n²) : désactivé au-delà de `max_docs`
    documents plutôt que de risquer un calcul trop long ou trop gourmand en
    mémoire sur un très gros lot."""
    n = len(paths)
    if n < 2 or n > max_docs:
        return []

    similarity = vectors @ vectors.T
    pairs: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity[i, j])
            if score >= threshold:
                pairs.append({
                    "path_a": paths[i], "path_b": paths[j],
                    "filename_a": filenames[i], "filename_b": filenames[j],
                    "similarity": round(score, 4),
                })
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs


def duplicate_removal_candidates(pairs: list[dict]) -> list[str]:
    """À partir des paires détectées, choisit pour chaque groupe de quasi-
    doublons UN fichier à garder (le premier rencontré) et retourne les
    chemins des autres, considérés "en trop". Une paire dont l'un des deux
    fichiers a déjà été retenu comme "en trop" par une paire précédente
    (groupe de plus de 2 quasi-doublons) est ignorée pour éviter de proposer
    deux fois le même fichier."""
    to_remove: list[str] = []
    removed: set[str] = set()
    for pair in pairs:
        path_a, path_b = pair["path_a"], pair["path_b"]
        if path_a in removed or path_b in removed:
            continue
        to_remove.append(path_b)
        removed.add(path_b)
    return to_remove


def move_files_to_local_backup(paths: list[str]) -> list[str]:
    """Comme `move_to_backup`, mais choisit pour CHAQUE fichier un dossier
    `_backup` à côté de SON PROPRE dossier d'origine plutôt qu'un unique
    dossier de secours partagé — les fichiers à mettre de côté (ex. doublons
    détectés) peuvent provenir de plusieurs dossiers différents, et il est
    plus intuitif de retrouver chacun juste à côté de là où il était."""
    by_parent: dict[str, list[str]] = {}
    for path in paths:
        parent = os.path.dirname(os.path.abspath(path))
        by_parent.setdefault(parent, []).append(path)
    moved: list[str] = []
    for parent, group in by_parent.items():
        moved.extend(move_to_backup(group, os.path.join(parent, "_backup")))
    return moved


def move_to_backup(paths: list[str], backup_dir: str) -> list[str]:
    """Déplace (jamais ne supprime) chaque fichier vers `backup_dir` — un
    suffixe numérique est ajouté en cas de collision de nom (voir
    `_unique_destination`). Réversible, contrairement à une suppression :
    c'est tout l'intérêt de ce dossier de secours plutôt qu'un nettoyage
    définitif immédiat. Retourne les chemins de destination des fichiers
    effectivement déplacés (les chemins déjà introuvables sont ignorés en
    silence, ex. fichier déjà déplacé par une action précédente)."""
    os.makedirs(backup_dir, exist_ok=True)
    moved = []
    for path in paths:
        if not os.path.exists(path):
            continue
        dest = _unique_destination(os.path.join(backup_dir, os.path.basename(path)))
        shutil.move(path, dest)
        moved.append(dest)
    return moved
