"""Renommage des catégories d'un modèle déjà entraîné (mode non supervisé).

Les noms de catégories détectés automatiquement (mots-clés TF-IDF) ne sont
pas toujours parlants — ce module permet de les remplacer par des noms
choisis par l'utilisateur, dans le modèle lui-même et dans son dossier
`dataset/` (voir `model_store.model_dataset_dir`), pour que les deux restent
cohérents. Le dossier et le fichier .json associés se déduisent uniquement
du chemin du modèle : aucun état à transmettre séparément.
"""
from __future__ import annotations

import json
import os
import re
import shutil

from . import model_store
from .config import get_config
from .utils import write_json_atomic

_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def rename_categories(model_path: str, renames: dict[str, str]) -> dict:
    """`renames` associe un ancien nom de catégorie à son nouveau nom.
    Renomme dans le modèle (`cluster_names`) et dans son dossier `dataset/`
    (sous-dossiers + fichier .json de référence)."""
    bundle = model_store.load_bundle(model_path)
    if bundle.get("mode") != "unsupervised":
        raise ValueError(
            "Seuls les modèles créés par l'onglet Entraînement (catégories détectées "
            "automatiquement) peuvent être renommés ici."
        )

    cluster_names = bundle["cluster_names"]
    for cluster_id, name in list(cluster_names.items()):
        new_name = renames.get(name)
        if new_name and new_name != name:
            cluster_names[cluster_id] = new_name
    bundle["cluster_names"] = cluster_names

    model_store.snapshot_model(model_path)
    model_store.save_bundle(bundle, model_path)

    dataset_dir = model_store.model_dataset_dir(model_path)
    if os.path.isdir(dataset_dir):
        _rename_dataset_folders(model_path, dataset_dir, renames)

    return bundle


def _rename_dataset_folders(model_path: str, dataset_dir: str, renames: dict[str, str]) -> None:
    for old_name, new_name in renames.items():
        if not new_name or new_name == old_name:
            continue
        old_path = os.path.join(dataset_dir, old_name)
        new_path = os.path.join(dataset_dir, new_name)
        if not os.path.isdir(old_path):
            continue
        if os.path.isdir(new_path):
            # Une catégorie du même nouveau nom existe déjà : on fusionne le contenu.
            for filename in os.listdir(old_path):
                shutil.move(os.path.join(old_path, filename), os.path.join(new_path, filename))
            os.rmdir(old_path)
        else:
            os.rename(old_path, new_path)

    manifest_path = model_store.model_manifest_path(model_path)
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["categories"] = {
        cid: renames.get(name, name) for cid, name in manifest.get("categories", {}).items()
    }
    files_entry = manifest.get("files", {})
    for entry in files_entry.values():
        if not isinstance(entry, dict):
            continue  # ancien format (chaîne simple) laissé par une version antérieure
        old_category = entry.get("category")
        new_category = renames.get(old_category, old_category)
        if new_category == old_category:
            continue
        entry["category"] = new_category
        # Le dossier a été renommé/fusionné physiquement ci-dessus : il faut
        # que le chemin enregistré suive, sinon le prochain ré-entraînement
        # croirait le fichier manquant et en recréerait une copie en double.
        dataset_path = entry.get("dataset_path")
        if dataset_path:
            entry["dataset_path"] = os.path.join(dataset_dir, new_category, os.path.basename(dataset_path))
    manifest["files"] = files_entry
    write_json_atomic(manifest, manifest_path)


def delete_category(model_path: str, category_name: str, other_name: str | None = None) -> dict:
    """"Supprime" une catégorie en la fusionnant dans la catégorie fourre-tout
    (`other_category_name` de la configuration, "autre" par défaut) : les
    documents ne sont jamais perdus, seulement regroupés ailleurs."""
    other_name = other_name or get_config().other_category_name
    if category_name == other_name:
        raise ValueError(f"La catégorie {other_name!r} ne peut pas être fusionnée avec elle-même.")
    return rename_categories(model_path, {category_name: other_name})


def list_category_files(dataset_dir: str, category: str) -> list[str]:
    """Fichiers présents dans le sous-dossier `dataset/<category>/`."""
    category_dir = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_dir):
        return []
    return sorted(f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f)))


def _sanitize_prefix(name: str) -> str:
    cleaned = _FORBIDDEN_FILENAME_CHARS.sub("_", name).strip()
    return cleaned or "categorie"


def rename_files_with_prefix(model_path: str, category: str) -> int:
    """Préfixe chaque fichier du sous-dossier d'une catégorie par le nom de
    cette catégorie (ex: 'facture1.pdf' → 'Factures_facture1.pdf'). N'affecte
    que les copies du dataset, jamais les documents d'origine. Idempotent :
    un fichier déjà préfixé n'est pas renommé une seconde fois. Met aussi à
    jour le fichier .json de référence (sinon le prochain ré-entraînement
    croirait ces fichiers manquants et en recréerait des copies non
    préfixées). Retourne le nombre de fichiers effectivement renommés."""
    dataset_dir = model_store.model_dataset_dir(model_path)
    category_dir = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_dir):
        return 0

    model_store.snapshot_model(model_path)

    prefix = _sanitize_prefix(category) + "_"
    renamed_paths: dict[str, str] = {}
    for filename in list_category_files(dataset_dir, category):
        if filename.startswith(prefix):
            continue
        new_name = prefix + filename
        old_path = os.path.join(category_dir, filename)
        new_path = os.path.join(category_dir, new_name)
        if os.path.exists(new_path):
            continue  # évite d'écraser un fichier déjà présent sous ce nom
        os.rename(old_path, new_path)
        renamed_paths[os.path.normpath(old_path)] = new_path

    if renamed_paths:
        _update_manifest_dataset_paths(model_path, renamed_paths)
    return len(renamed_paths)


def _update_manifest_dataset_paths(model_path: str, renamed_paths: dict[str, str]) -> None:
    manifest_path = model_store.model_manifest_path(model_path)
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    files_entry = manifest.get("files", {})
    for entry in files_entry.values():
        if not isinstance(entry, dict):
            continue
        current = entry.get("dataset_path")
        if current and os.path.normpath(current) in renamed_paths:
            entry["dataset_path"] = renamed_paths[os.path.normpath(current)]
    manifest["files"] = files_entry
    write_json_atomic(manifest, manifest_path)
