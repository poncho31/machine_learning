"""Sauvegarde/chargement atomique des modèles entraînés (.pkl), et résolution
de la structure de dossiers propre à chaque modèle :

    storage/models/<nom>/
        <nom>.pkl              modèle entraîné
        <nom>.json               catégories + fichiers qui y contribuent
        dataset/                   documents utilisés pour le construire et l'améliorer,
          <catégorie>/               organisés par catégorie
        pkl_history/<nom>_<horodatage>.pkl        instantanés précédents du modèle
        json_history/<nom>_<horodatage>.json        instantanés précédents du .json
        dataset_history/<horodatage>/                 instantanés précédents du dataset

Cette structure ne dépend d'aucun état volatile (rien dans le bundle) : elle
se retrouve simplement à partir du chemin du modèle, y compris après un
redémarrage de l'application — c'est ce qui garantit qu'on ne perd jamais
l'accès aux fichiers déjà catégorisés. Un instantané est pris avant toute
modification (entraînement/amélioration/renommage) : voir `snapshot_model`
et `restore_snapshot`.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import tempfile
from datetime import datetime

from .config import get_config
from .utils import write_json_atomic

_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*]')
_HISTORY_DIR_NAMES = ("pkl_history", "json_history", "dataset_history")


def sanitize_model_name(name: str) -> str:
    cleaned = _UNSAFE_NAME_CHARS.sub("_", name).strip()
    return cleaned or "modele"


def model_path_for_name(name: str, root: str | None = None) -> str:
    """Chemin canonique du modèle `name` : storage/models/<name>/<name>.pkl
    (ou sous `root` si fourni, par défaut la valeur configurée)."""
    root = root if root is not None else get_config().models_root
    safe_name = sanitize_model_name(name)
    return os.path.join(root, safe_name, f"{safe_name}.pkl")


def model_dataset_dir(model_path: str) -> str:
    """Dossier des documents ayant contribué à ce modèle, à côté du .pkl."""
    return os.path.join(os.path.dirname(os.path.abspath(model_path)) or ".", "dataset")


def model_manifest_path(model_path: str) -> str:
    """Fichier .json référençant les catégories et fichiers de ce modèle,
    à côté du .pkl (même nom, extension .json)."""
    directory = os.path.dirname(os.path.abspath(model_path)) or "."
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(directory, f"{stem}.json")


def load_manifest(model_path: str) -> dict:
    """Charge le manifeste (`<nom>.json`) d'un modèle — un dict vide si le
    fichier n'existe pas encore (modèle tout juste créé, ou jamais
    synchronisé) ou est illisible (JSON corrompu, erreur disque), plutôt que
    de lever : cette même garde était auparavant recopiée telle quelle dans
    plus d'une dizaine d'endroits (discover.py, rename.py, gui.py) à chaque
    fois qu'un appelant avait besoin de lire ce fichier."""
    manifest_path = model_manifest_path(model_path)
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_manifest(manifest: dict, model_path: str) -> None:
    write_json_atomic(manifest, model_manifest_path(model_path))


def model_digest_path(model_path: str) -> str:
    """Emplacement de l'ANCIEN résumé de corpus séparé (`_digest.json`),
    d'avant son unification dans `<nom>.json` (voir `model_manifest_path`,
    `discover._write_corpus_digest`) : un seul fichier fait désormais
    référence pour un modèle donné (catégories, doublons, résumé par
    document), plutôt que deux qui pouvaient se désynchroniser. Cette
    fonction n'est plus utilisée que pour repérer et supprimer un éventuel
    `_digest.json` laissé par une version antérieure de l'application."""
    directory = os.path.dirname(os.path.abspath(model_path)) or "."
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(directory, f"{stem}_digest.json")


def _model_dir(model_path: str) -> str:
    return os.path.dirname(os.path.abspath(model_path)) or "."


def _model_stem(model_path: str) -> str:
    return os.path.splitext(os.path.basename(model_path))[0]


def snapshot_model(model_path: str) -> str | None:
    """Archive l'état courant du modèle (.pkl, .json, dataset/) avant de le
    modifier, pour pouvoir revenir en arrière. N'archive que ce qui existe
    déjà — un modèle tout juste créé n'a rien à archiver. Retourne
    l'horodatage de l'instantané, ou None si rien n'a été archivé.
    Respecte `model_history_keep` (0 = historique désactivé) en supprimant
    les plus anciens instantanés au-delà de cette limite."""
    if get_config().model_history_keep <= 0:
        return None
    if not os.path.exists(model_path):
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    model_dir = _model_dir(model_path)
    stem = _model_stem(model_path)

    pkl_history_dir = os.path.join(model_dir, "pkl_history")
    os.makedirs(pkl_history_dir, exist_ok=True)
    shutil.copy2(model_path, os.path.join(pkl_history_dir, f"{stem}_{timestamp}.pkl"))

    manifest_path = model_manifest_path(model_path)
    if os.path.exists(manifest_path):
        json_history_dir = os.path.join(model_dir, "json_history")
        os.makedirs(json_history_dir, exist_ok=True)
        shutil.copy2(manifest_path, os.path.join(json_history_dir, f"{stem}_{timestamp}.json"))

    dataset_dir = model_dataset_dir(model_path)
    if os.path.isdir(dataset_dir):
        dataset_history_root = os.path.join(model_dir, "dataset_history")
        os.makedirs(dataset_history_root, exist_ok=True)
        shutil.copytree(dataset_dir, os.path.join(dataset_history_root, timestamp))

    _prune_history(model_path)
    return timestamp


def _prune_history(model_path: str) -> None:
    keep = get_config().model_history_keep
    model_dir = _model_dir(model_path)
    for timestamp in list_snapshots(model_path)[keep:]:
        _remove_snapshot_files(model_path, timestamp)
    # Répertoires d'historique jamais créés tant qu'aucun instantané n'existe.
    for name in _HISTORY_DIR_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)


def _remove_snapshot_files(model_path: str, timestamp: str) -> None:
    model_dir = _model_dir(model_path)
    stem = _model_stem(model_path)
    pkl_path = os.path.join(model_dir, "pkl_history", f"{stem}_{timestamp}.pkl")
    json_path = os.path.join(model_dir, "json_history", f"{stem}_{timestamp}.json")
    dataset_path = os.path.join(model_dir, "dataset_history", timestamp)
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    if os.path.exists(json_path):
        os.remove(json_path)
    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path, ignore_errors=True)


def list_snapshots(model_path: str) -> list[str]:
    """Horodatages des instantanés disponibles pour ce modèle, du plus
    récent au plus ancien."""
    model_dir = _model_dir(model_path)
    stem = _model_stem(model_path)
    pkl_history_dir = os.path.join(model_dir, "pkl_history")
    if not os.path.isdir(pkl_history_dir):
        return []
    prefix, suffix = f"{stem}_", ".pkl"
    timestamps = [
        filename[len(prefix):-len(suffix)]
        for filename in os.listdir(pkl_history_dir)
        if filename.startswith(prefix) and filename.endswith(suffix)
    ]
    return sorted(timestamps, reverse=True)


def restore_snapshot(model_path: str, timestamp: str) -> None:
    """Restaure le modèle (.pkl, .json, dataset/) à l'état d'un instantané
    précédent. L'état courant est d'abord archivé à son tour (comme toute
    autre modification) : restaurer peut donc lui-même être annulé."""
    model_dir = _model_dir(model_path)
    stem = _model_stem(model_path)
    pkl_snapshot = os.path.join(model_dir, "pkl_history", f"{stem}_{timestamp}.pkl")
    if not os.path.exists(pkl_snapshot):
        raise FileNotFoundError(f"Instantané introuvable : {timestamp}")

    snapshot_model(model_path)  # archive l'état courant avant de l'écraser

    shutil.copy2(pkl_snapshot, model_path)

    json_snapshot = os.path.join(model_dir, "json_history", f"{stem}_{timestamp}.json")
    manifest_path = model_manifest_path(model_path)
    if os.path.exists(json_snapshot):
        shutil.copy2(json_snapshot, manifest_path)
    elif os.path.exists(manifest_path):
        os.remove(manifest_path)

    dataset_snapshot = os.path.join(model_dir, "dataset_history", timestamp)
    dataset_dir = model_dataset_dir(model_path)
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.isdir(dataset_snapshot):
        shutil.copytree(dataset_snapshot, dataset_dir)


def delete_model_permanently(model_path: str) -> None:
    """Supprime DÉFINITIVEMENT ce modèle : le `.pkl`, son `.json`, son
    dossier `dataset/`, et tout son historique (pkl_history/json_history/
    dataset_history) — AUCUN instantané n'est pris avant (il n'y aurait nulle
    part où le restaurer une fois ces fichiers eux-mêmes supprimés).

    Ne supprime QUE ces chemins précis, jamais tout le dossier qui les
    contient : un modèle créé via "Parcourir..." plutôt que par le nom
    standard (voir `model_path_for_name`) peut très bien partager son
    dossier avec des fichiers sans rapport (ex. le Bureau de l'utilisateur)
    — les y laisser intacts est le seul comportement sûr. Ne touche jamais
    aux documents SOURCE : `dataset/` n'en est qu'une copie."""
    if os.path.exists(model_path):
        os.remove(model_path)
    manifest_path = model_manifest_path(model_path)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    legacy_digest = model_digest_path(model_path)
    if os.path.exists(legacy_digest):
        os.remove(legacy_digest)
    dataset_dir = model_dataset_dir(model_path)
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    model_dir = _model_dir(model_path)
    for name in _HISTORY_DIR_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def save_bundle(bundle: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(bundle, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_bundle(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


_IGNORED_DIR_NAMES = {"__pycache__", "venv", "node_modules", *_HISTORY_DIR_NAMES}


def discover_models(root: str = ".", max_depth: int | None = None) -> list[tuple[str, int]]:
    """Cherche les fichiers .pkl sous `root` (dossiers cachés et environnements
    virtuels ignorés), triés du plus léger au plus lourd. Sert à proposer les
    modèles déjà entraînés sans devoir les chercher à la main."""
    if max_depth is None:
        max_depth = get_config().model_discovery_max_depth
    root = os.path.abspath(root)
    results: list[tuple[str, int]] = []
    for current_root, dirs, files in os.walk(root):
        depth = os.path.relpath(current_root, root).count(os.sep) if current_root != root else 0
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _IGNORED_DIR_NAMES]
        for filename in files:
            if filename.lower().endswith(".pkl"):
                full_path = os.path.join(current_root, filename)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                results.append((full_path, size))
    results.sort(key=lambda item: item[1])
    return results
