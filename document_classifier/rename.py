"""Renommage des catégories d'un modèle déjà entraîné (mode non supervisé).

Les noms de catégories détectés automatiquement (mots-clés TF-IDF) ne sont
pas toujours parlants — ce module permet de les remplacer par des noms
choisis par l'utilisateur, dans le modèle lui-même et dans son dossier
`dataset/` (voir `model_store.model_dataset_dir`), pour que les deux restent
cohérents. Le dossier et le fichier .json associés se déduisent uniquement
du chemin du modèle : aucun état à transmettre séparément.
"""
from __future__ import annotations

import os
import re
import shutil

import numpy as np

from . import model_store
from .config import get_config
from .discover import detected_category_for_document
from .extraction import extract_text
from .features import ENGINE_IMAGE, engine_from_state
from .utils import content_hash, dispatch_file

_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def _require_unsupervised(bundle: dict) -> None:
    """Garde commune à toutes les opérations de ce module (renommer,
    fusionner, déplacer ou ajouter des fichiers à une catégorie) : n'a de
    sens que pour un modèle "non supervisé" (catégories détectées
    automatiquement, onglet Entraînement) — un modèle supervisé (CLI
    `train`) a ses catégories définies une fois pour toutes par
    l'utilisateur via des sous-dossiers, sans notion de cluster K-Means à
    renommer, fusionner ou réorganiser."""
    if bundle.get("mode") != "unsupervised":
        raise ValueError(
            "Seuls les modèles créés par l'onglet Entraînement (catégories détectées "
            "automatiquement) peuvent être modifiés ici."
        )


def _save_manifest_and_bundle(
    model_path: str, manifest: dict, files_entry: dict, bundle: dict, confirmed_overrides: dict
) -> None:
    """Enregistre le manifeste et le bundle après une modification directe de
    `files_entry`/`confirmed_overrides` — partagé par `move_files_to_category`
    et `add_files_to_category`. L'instantané (voir `model_store.snapshot_model`)
    doit déjà avoir été pris par l'appelant AVANT toute modification sur
    disque (copie/déplacement de fichiers, ci-dessus) pour qu'un retour en
    arrière restaure effectivement l'état précédent, pas un état déjà
    partiellement modifié."""
    manifest["files"] = files_entry
    model_store.save_manifest(manifest, model_path)
    bundle["confirmed_overrides"] = confirmed_overrides
    model_store.save_bundle(bundle, model_path)


def rename_categories(model_path: str, renames: dict[str, str]) -> dict:
    """`renames` associe un ancien nom de catégorie à son nouveau nom.
    Renomme dans le modèle (`cluster_names`) et dans son dossier `dataset/`
    (sous-dossiers + fichier .json de référence)."""
    bundle = model_store.load_bundle(model_path)
    _require_unsupervised(bundle)

    cluster_names = bundle["cluster_names"]
    for cluster_id, name in list(cluster_names.items()):
        new_name = renames.get(name)
        if new_name and new_name != name:
            cluster_names[cluster_id] = new_name
    bundle["cluster_names"] = cluster_names

    # Les catégories confirmées à la main (voir `discover._merge_confirmed_overrides`)
    # ne passent pas forcément par `cluster_names` (une correction peut créer
    # une catégorie qu'aucun cluster K-Means n'a nommée) : sans cette mise à
    # jour, un renommage ou une suppression ici serait silencieusement annulé
    # à la prochaine classification ou amélioration du modèle.
    confirmed_overrides = bundle.get("confirmed_overrides")
    if confirmed_overrides:
        bundle["confirmed_overrides"] = {
            digest: renames.get(name, name) for digest, name in confirmed_overrides.items()
        }

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

    manifest = model_store.load_manifest(model_path)
    if not manifest:
        return
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
    model_store.save_manifest(manifest, model_path)


def delete_category(model_path: str, category_name: str, other_name: str | None = None) -> dict:
    """"Supprime" une catégorie en la fusionnant dans la catégorie fourre-tout
    (`other_category_name` de la configuration, "autre" par défaut) : les
    documents ne sont jamais perdus, seulement regroupés ailleurs."""
    other_name = other_name or get_config().other_category_name
    if category_name == other_name:
        raise ValueError(f"La catégorie {other_name!r} ne peut pas être fusionnée avec elle-même.")
    return rename_categories(model_path, {category_name: other_name})


def delete_category_permanently(model_path: str, category_name: str) -> int:
    """Supprime DÉFINITIVEMENT une catégorie — contrairement à
    `delete_category` ci-dessus, qui la fusionne dans "autre" sans jamais
    perdre de document. Retire ses copies de `dataset/`, ses entrées du
    manifest et toute référence à elle dans le bundle (`cluster_names`,
    `original_cluster_names`, `confirmed_overrides`).

    Ne touche JAMAIS aux documents SOURCE : `dataset/` n'en est qu'une copie
    (voir `model_store.model_dataset_dir`) — si les fichiers d'origine
    existent toujours ailleurs sur le disque, un futur (ré)entraînement
    complet peut les redétecter et recréer une catégorie similaire. Cette
    suppression porte sur l'état ACTUEL du modèle, pas sur les documents
    eux-mêmes. Un instantané est pris avant, comme pour toute autre
    opération de ce module (voir `model_store.snapshot_model`) — irréversible
    seulement au sens où aucun "autre" fourre-tout ne récupère les documents,
    pas au sens où le modèle ne peut plus revenir en arrière. Retourne le
    nombre de fichiers supprimés."""
    bundle = model_store.load_bundle(model_path)
    _require_unsupervised(bundle)

    dataset_dir = model_store.model_dataset_dir(model_path)
    category_dir = os.path.join(dataset_dir, category_name)
    filenames = list_category_files(dataset_dir, category_name)
    if not filenames and not os.path.isdir(category_dir):
        return 0

    model_store.snapshot_model(model_path)

    if os.path.isdir(category_dir):
        shutil.rmtree(category_dir, ignore_errors=True)

    manifest = model_store.load_manifest(model_path)
    files_entry = manifest.get("files", {})
    for source_path, entry in list(files_entry.items()):
        if isinstance(entry, dict) and entry.get("category") == category_name:
            del files_entry[source_path]
    manifest["files"] = files_entry
    model_store.save_manifest(manifest, model_path)

    cluster_names = dict(bundle.get("cluster_names", {}))
    for cluster_id, name in list(cluster_names.items()):
        if name == category_name:
            del cluster_names[cluster_id]
    bundle["cluster_names"] = cluster_names
    original_names = dict(bundle.get("original_cluster_names", {}))
    for cluster_id in list(original_names):
        if cluster_id not in cluster_names:
            del original_names[cluster_id]
    bundle["original_cluster_names"] = original_names
    bundle["confirmed_overrides"] = {
        digest: name for digest, name in bundle.get("confirmed_overrides", {}).items() if name != category_name
    }
    model_store.save_bundle(bundle, model_path)
    return len(filenames)


def delete_file_from_category(model_path: str, category_name: str, filename: str) -> bool:
    """Supprime DÉFINITIVEMENT la copie `dataset/` d'UN fichier précis — sans
    le faire passer par "autre" (contrairement à un renommage/déplacement).
    Ne touche jamais le fichier SOURCE, seulement sa copie dans `dataset/`
    (mêmes réserves que `delete_category_permanently` : un futur
    ré-entraînement complet peut le redétecter si l'original existe
    toujours). Retourne False si le fichier n'existe pas (déjà supprimé,
    ou catégorie/nom incorrect)."""
    bundle = model_store.load_bundle(model_path)
    _require_unsupervised(bundle)

    dataset_dir = model_store.model_dataset_dir(model_path)
    file_path = os.path.join(dataset_dir, category_name, filename)
    if not os.path.isfile(file_path):
        return False

    text, _error = extract_text(file_path)

    model_store.snapshot_model(model_path)
    os.remove(file_path)

    manifest = model_store.load_manifest(model_path)
    files_entry = manifest.get("files", {})
    normalized_target = os.path.normpath(file_path)
    for source_path, entry in list(files_entry.items()):
        if isinstance(entry, dict) and os.path.normpath(entry.get("dataset_path", "")) == normalized_target:
            del files_entry[source_path]
    manifest["files"] = files_entry
    model_store.save_manifest(manifest, model_path)

    if text.strip():
        digest = content_hash(text)
        confirmed_overrides = bundle.get("confirmed_overrides", {})
        if digest in confirmed_overrides:
            bundle["confirmed_overrides"] = {k: v for k, v in confirmed_overrides.items() if k != digest}
            model_store.save_bundle(bundle, model_path)
    return True


def move_files_to_category(model_path: str, filenames: list[str], from_category: str, to_category: str) -> int:
    """Déplace des fichiers précis (choisis dans la liste des fichiers d'UNE
    catégorie, dans la section "Catégories de ce modèle" de l'onglet
    Entraînement — sélection multiple possible) de `from_category` vers
    `to_category`, dans le dossier `dataset/` du modèle.

    Contrairement à `rename_categories` (qui renomme une catégorie ENTIÈRE),
    ce choix porte sur des documents précis : il est donc mémorisé comme une
    correction confirmée à la main (empreinte du contenu, voir
    `discover._merge_confirmed_overrides`), pour qu'un futur ré-entraînement
    (bouton "Améliorer le modèle" de l'onglet Classification, ou un nouvel
    entraînement basé sur ce modèle) ne le fasse pas silencieusement revenir
    à sa catégorie d'origine au prochain passage de regroupement K-Means."""
    bundle = model_store.load_bundle(model_path)
    _require_unsupervised(bundle)
    if from_category == to_category or not filenames:
        return 0

    # AVANT toute modification sur disque (déplacement de fichiers ci-dessous) :
    # un instantané pris plus tard, une fois les fichiers déjà déplacés,
    # archiverait un dataset/ déjà modifié — inutile pour revenir en arrière.
    model_store.snapshot_model(model_path)

    dataset_dir = model_store.model_dataset_dir(model_path)
    from_dir = os.path.join(dataset_dir, from_category)

    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})
    dataset_path_to_source = {
        os.path.normpath(entry["dataset_path"]): source
        for source, entry in files_entry.items()
        if isinstance(entry, dict) and entry.get("dataset_path")
    }

    engine = engine_from_state(bundle["engine_state"])
    confirmed_overrides: dict[str, str] = dict(bundle.get("confirmed_overrides", {}))
    moved = 0
    for filename in filenames:
        src = os.path.join(from_dir, filename)
        if not os.path.isfile(src):
            continue
        dest = dispatch_file(src, to_category, dataset_dir, move=True)
        moved += 1

        text, _error = extract_text(dest)
        source_path = dataset_path_to_source.get(os.path.normpath(src))
        if source_path:
            # Fusionne plutôt qu'écrase : préserve char_count/keywords/excerpt
            # déjà enregistrés pour ce fichier (voir discover._write_corpus_digest)
            # au lieu de les perdre. `detected_category` (mots-clés dominants
            # DU DOCUMENT LUI-MÊME, TOUJOURS renseigné — jamais "(confirmée
            # manuellement)", voir `discover.detected_category_for_document`)
            # est recalculé à chaque déplacement plutôt que conservé de
            # l'ancienne catégorie.
            previous_entry = files_entry.get(source_path, {})
            entry = {**previous_entry, "category": to_category, "dataset_path": dest}
            if text.strip():
                entry["detected_category"] = detected_category_for_document(bundle, engine, text)
            files_entry[source_path] = entry

        if text.strip():
            confirmed_overrides[content_hash(text)] = to_category

    # `from_dir` peut se retrouver vide une fois tous ses fichiers déplacés
    # (sélection complète plutôt que partielle) — le nettoyer plutôt que de
    # le laisser traîner sur le disque : sans ça, ce dossier vide resterait
    # visible comme une catégorie "fantôme" pour tout code qui énumère les
    # sous-dossiers de dataset/ (voir `compute_dataset_vectors`), même si
    # `bundle["cluster_names"]` ne le mentionne plus ici.
    if moved and os.path.isdir(from_dir) and not os.listdir(from_dir):
        try:
            os.rmdir(from_dir)
        except OSError:
            pass

    if moved:
        _save_manifest_and_bundle(model_path, manifest, files_entry, bundle, confirmed_overrides)
    return moved


def add_files_to_category(model_path: str, file_paths: list[str], category: str) -> int:
    """Ajoute des fichiers PRÉCIS (choisis n'importe où sur le disque, pas
    forcément déjà connus du modèle) directement dans une catégorie donnée —
    l'opération inverse de `move_files_to_category` : au lieu de partir d'un
    fichier déjà classé pour lui choisir sa catégorie, on part d'une
    catégorie choisie pour lui affecter directement un ou plusieurs
    fichiers, sans passer par la prédiction de l'onglet Classification.

    Comme un déplacement, ce choix est mémorisé comme une correction
    confirmée à la main (empreinte du contenu, voir
    `discover._merge_confirmed_overrides`), pour qu'un futur ré-entraînement
    ou une future classification le reprenne plutôt que de le laisser au
    hasard du regroupement K-Means."""
    bundle = model_store.load_bundle(model_path)
    _require_unsupervised(bundle)
    if not category or not file_paths:
        return 0

    # AVANT toute modification sur disque (copie de fichiers ci-dessous),
    # même raison que dans `move_files_to_category`.
    model_store.snapshot_model(model_path)

    dataset_dir = model_store.model_dataset_dir(model_path)
    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})

    engine = engine_from_state(bundle["engine_state"])
    confirmed_overrides: dict[str, str] = dict(bundle.get("confirmed_overrides", {}))
    added = 0
    for path in file_paths:
        if not os.path.isfile(path):
            continue
        dest = dispatch_file(path, category, dataset_dir, move=False)
        text, _error = extract_text(dest)
        entry = {**files_entry.get(path, {}), "category": category, "dataset_path": dest}
        if text.strip():
            # `detected_category` (mots-clés dominants DU DOCUMENT LUI-MÊME,
            # TOUJOURS renseigné — jamais "(confirmée manuellement)", voir
            # `discover.detected_category_for_document`) : ce fichier n'est
            # jamais passé par la prédiction de l'onglet Classification, donc
            # il faut l'analyser ici.
            entry["detected_category"] = detected_category_for_document(bundle, engine, text)
        files_entry[path] = entry
        added += 1

        if text.strip():
            confirmed_overrides[content_hash(text)] = category

    if added:
        _save_manifest_and_bundle(model_path, manifest, files_entry, bundle, confirmed_overrides)
    return added


def list_category_files(dataset_dir: str, category: str) -> list[str]:
    """Fichiers présents dans le sous-dossier `dataset/<category>/`."""
    category_dir = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_dir):
        return []
    return sorted(f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f)))


def compute_dataset_vectors(bundle: dict, dataset_dir: str, progress=None) -> tuple[np.ndarray, list[str]]:
    """Revectorise, avec le moteur déjà entraîné de `bundle` (aucun
    réajustement, seulement `.transform()`), tous les documents actuellement
    présents dans `dataset/<catégorie>/` — reflète donc l'état RÉEL et
    ACTUEL du modèle (renommages, fusions, ajouts/déplacements manuels
    compris), plutôt qu'un instantané figé au moment d'un entraînement
    précédent. Utilisé par l'aperçu du regroupement (2D) de l'onglet
    Entraînement : peut être appelé sur N'IMPORTE QUEL modèle non supervisé
    déjà entraîné, qu'il vienne d'être (ré)entraîné ou simplement rechargé
    depuis le disque.

    Chaque nom de catégorie renvoyé est directement le nom du sous-dossier
    `dataset/` d'où vient le document — jamais un identifiant de cluster
    interne à retraduire en nom affiché, ce qui exclut par construction tout
    décalage avec ce qu'affiche la section "Catégories de ce modèle".

    Seuls les sous-dossiers correspondant à une catégorie ACTUELLEMENT
    reconnue par le modèle (`cluster_names`, plus les catégories confirmées
    à la main via `confirmed_overrides` — exactement l'ensemble que calcule
    `gui._populate_categories`, `by_name`) sont pris en compte : un
    sous-dossier resté sur le disque après un renommage/déplacement (ex. un
    dossier vidé de tous ses fichiers par "Déplacer la sélection vers" mais
    pas encore nettoyé) ne doit pas ressurgir comme une fausse catégorie
    "ancienne" dans l'aperçu."""
    if not os.path.isdir(dataset_dir):
        return np.empty((0, 0)), []

    current_categories = set(bundle.get("cluster_names", {}).values()) | set(
        bundle.get("confirmed_overrides", {}).values()
    )
    categories = sorted(
        name for name in os.listdir(dataset_dir)
        if name in current_categories and os.path.isdir(os.path.join(dataset_dir, name))
    )
    # Le moteur "image" (voir discover._build_bundle) vectorise le contenu
    # visuel du fichier, pas du texte extrait — inutile (et vide sans OCR)
    # de tenter une extraction de texte pour lui, seul le chemin compte.
    is_image_engine = bundle.get("engine_state", {}).get("type") == ENGINE_IMAGE
    vectorize_inputs: list[str] = []
    labels: list[str] = []
    for category in categories:
        filenames = list_category_files(dataset_dir, category)
        read_count = 0
        for filename in filenames:
            path = os.path.join(dataset_dir, category, filename)
            if is_image_engine:
                vectorize_inputs.append(path)
            else:
                text, _error = extract_text(path)
                if not text.strip():
                    continue
                vectorize_inputs.append(text)
            labels.append(category)
            read_count += 1
        if progress:
            progress(f"  {category} : {read_count}/{len(filenames)} fichier(s) relu(s)...")

    if not vectorize_inputs:
        return np.empty((0, 0)), []

    if progress:
        progress(f"Vectorisation de {len(vectorize_inputs)} document(s)...")
    engine = engine_from_state(bundle["engine_state"])
    vectors = engine.transform(vectorize_inputs)
    return vectors, labels


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
    manifest = model_store.load_manifest(model_path)
    if not manifest:
        return
    files_entry = manifest.get("files", {})
    for entry in files_entry.values():
        if not isinstance(entry, dict):
            continue
        current = entry.get("dataset_path")
        if current and os.path.normpath(current) in renamed_paths:
            entry["dataset_path"] = renamed_paths[os.path.normpath(current)]
    manifest["files"] = files_entry
    model_store.save_manifest(manifest, model_path)
