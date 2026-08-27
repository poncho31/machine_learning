"""Mode découverte : regroupe des documents non triés par similarité (non
supervisé) — aucune organisation manuelle préalable n'est nécessaire.

Le nombre de catégories est détecté automatiquement (via le score de
silhouette) plutôt que fixé en dur, et chaque catégorie reçoit un nom lisible
basé sur ses mots-clés les plus représentatifs (français + anglais). Si aucun
découpage testé n'atteint un score de silhouette suffisant (voir la config
`cluster_min_silhouette`) ni un nombre de documents minimal par catégorie
(`cluster_min_cluster_size`), une seule catégorie est gardée plutôt que d'en
forcer plusieurs arbitrairement (voir `_best_k`).

Le score de silhouette n'a PAS la même échelle selon le moteur : TF-IDF,
dans son espace creux et de grande dimension, produit des scores
structurellement plus bas que les embeddings, même pour une séparation
réellement correcte (mesuré : ~0.03 pour TF-IDF contre ~0.19 pour les
embeddings sur le MÊME découpage correct d'un même corpus de test). Un seuil
de silhouette unique ne peut donc pas parfaitement distinguer signal et bruit
pour les deux moteurs à la fois — voir la config `cluster_min_silhouette` et
les préréglages de l'onglet Entraînement, qui permettent d'ajuster ce
compromis au cas par cas plutôt que de dépendre d'une seule valeur globale.

Deux points d'entrée :
- `discover` : range aussi les fichiers dans des sous-dossiers (usage CLI).
- `build_model` : construit le modèle (utilisé par l'onglet Entraînement de
  la GUI), avec la possibilité de fusionner les documents d'un modèle
  existant pour l'améliorer. Ne déplace jamais les documents d'origine,
  mais tient à jour une copie dans le dossier `dataset/` propre au modèle
  (voir `model_store.model_dataset_dir`) — c'est ce dossier, stable et
  déductible du seul chemin du modèle, que consulte l'onglet Transformer
  les données, y compris après un redémarrage de l'application.
- `improve_model` : même chose, pour l'amélioration continue depuis
  l'onglet Classification (voir ce module).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from . import model_store
from .classify import load_model_for_prediction, predict_labels, unreadable_category
from .config import get_config
from .extraction import ExtractedDocument, extract_documents
from .features import STOPWORDS, TOKEN_PATTERN, ENGINE_TFIDF, TfidfEngine, create_engine, engine_to_state
from .utils import dispatch_file, write_json_atomic


def _best_k(
    vectors: np.ndarray,
    k_min: int,
    k_max: int,
    min_silhouette: float | None = None,
    min_cluster_size: int | None = None,
) -> int:
    """Cherche, parmi k_min..k_max, le nombre de catégories qui maximise le
    score de silhouette — mais seulement si ce meilleur score dépasse
    `min_silhouette` (config `cluster_min_silhouette`). En dessous, aucune
    valeur de k testée ne correspond à une séparation réelle entre
    documents : mieux vaut garder tout le lot en une seule catégorie
    (k=1) que de forcer un découpage qui ne reflète que du bruit (ex. un
    dossier ne contenant qu'un seul type de document).

    Un k dont au moins un cluster contient moins de `min_cluster_size`
    documents (config `cluster_min_cluster_size`) est écarté avant même de
    regarder son score : sur un petit corpus, le score de silhouette grimpe
    sinon artificiellement à mesure que k s'approche du nombre total de
    documents, un cluster à un seul document ayant toujours une cohésion
    parfaite — sans que cela corresponde à une vraie catégorie."""
    if min_silhouette is None:
        min_silhouette = get_config().cluster_min_silhouette
    if min_cluster_size is None:
        min_cluster_size = get_config().cluster_min_cluster_size

    candidates = range(k_min, k_max + 1) if k_max > k_min else [k_min]
    best_k, best_score = k_min, -1.0
    for k in candidates:
        if k <= 1:
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        if np.bincount(labels).min() < min_cluster_size:
            continue
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k, best_score = k, score

    if best_score < min_silhouette:
        return 1
    return best_k


def _boilerplate_terms(texts: list[str], doc_freq_threshold: float = 0.8) -> set[str]:
    """Termes présents dans une trop grande proportion des documents SOURCE
    (adresse, nom de l'employeur, nom du destinataire...) : jamais
    discriminants pour nommer une catégorie, mais peuvent quand même
    dominer le nommage une fois les clusters devenus petits — leur IDF,
    recalculée seulement sur la poignée de clusters à nommer, ne suffit
    alors plus à les écarter. Calculé une fois sur le corpus complet,
    indépendamment du découpage en clusters."""
    if len(texts) < 2:
        return set()
    vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words=STOPWORDS, binary=True)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return set()
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel() / len(texts)
    terms = vectorizer.get_feature_names_out()
    return {terms[i] for i in range(len(terms)) if doc_freq[i] >= doc_freq_threshold}


def _name_clusters(
    texts: list[str],
    labels: np.ndarray,
    n_clusters: int,
    top_n: int = 3,
    previous_names: list[str | None] | None = None,
) -> tuple[dict[int, str], set[int], dict[int, str]]:
    """Nomme chaque cluster. Si `previous_names[i]` donne la catégorie que le
    document `i` avait dans un modèle précédent (renommages/suppressions
    compris — voir `_infer_previous_labels`), le nom majoritaire des
    documents d'un cluster est repris tel quel plutôt que régénéré : c'est
    ce qui permet à un ré-entraînement de ne pas perdre les catégories déjà
    renommées ou supprimées dans l'onglet Transformer les données.

    Le nom "brut" basé sur les mots-clés TF-IDF est toujours calculé pour
    chaque cluster, même quand un nom repris est utilisé comme nom affiché —
    c'est ce nom brut, jamais écrasé par un renommage manuel, qu'affiche
    l'onglet Transformer les données comme "nom détecté par le modèle"."""
    names: dict[int, str] = {}
    carried_over: set[int] = set()

    if previous_names is not None:
        for cluster_id in range(n_clusters):
            votes = [
                previous_names[i]
                for i in range(len(labels))
                if labels[i] == cluster_id and previous_names[i]
            ]
            if not votes:
                continue
            majority_name, count = Counter(votes).most_common(1)[0]
            if count / len(votes) >= 0.5:
                names[cluster_id] = majority_name
                carried_over.add(cluster_id)

    cluster_texts = [""] * n_clusters
    for text, label in zip(texts, labels):
        cluster_texts[label] += " " + text

    # Écarte les mots présents dans la quasi-totalité des documents SOURCE
    # (adresse, nom de l'employeur, nom du destinataire...) : avec beaucoup
    # de petits clusters, leur IDF locale ne suffit plus à les distinguer
    # d'un vrai mot-clé de catégorie (voir _boilerplate_terms).
    boilerplate = _boilerplate_terms(texts)

    if any(t.strip() for t in cluster_texts):
        naming_engine = TfidfEngine(max_features=1000)
        matrix = naming_engine.fit_transform(cluster_texts)
        terms = naming_engine.vectorizer.get_feature_names_out()
    else:
        matrix, terms = None, None

    raw_names: dict[int, str] = {}
    for i in range(n_clusters):
        name = None
        if matrix is not None:
            row = matrix[i]
            if row.any():
                order = row.argsort()[::-1]
                words = []
                for j in order:
                    if row[j] <= 0:
                        break
                    if terms[j] in boilerplate:
                        continue
                    words.append(terms[j])
                    if len(words) >= top_n:
                        break
                if words:
                    name = "_".join(words)
        raw_names[i] = name if name is not None else f"categorie_{i + 1}"

    used: set[str] = set(names.values())
    for i in range(n_clusters):
        if i in carried_over:
            continue
        name = raw_names[i]
        base_name, suffix = name, 2
        while name in used:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used.add(name)
        names[i] = name

    return names, carried_over, raw_names


def _infer_previous_labels(base_model_path: str, documents: list[ExtractedDocument], progress) -> dict[str, str]:
    """Fait repasser des documents dans un modèle déjà entraîné pour
    connaître la catégorie qu'il leur attribuerait — y compris les
    renommages et les suppressions (fusionnées dans "autre") faits dans
    l'onglet Transformer les données, puisque `predict_labels` lit
    `cluster_names` tel qu'il est actuellement enregistré dans le modèle.

    Seules les prédictions suffisamment confiantes (même seuil que pour la
    classification) comptent comme "vote" : un document sur lequel l'ancien
    modèle est incertain ne doit pas imposer un ancien nom à un cluster qui
    correspond en réalité à un sujet nouveau."""
    readable = [d for d in documents if not d.is_empty]
    if not readable:
        return {}
    try:
        base_bundle, base_engine = load_model_for_prediction(base_model_path)
        vectors = base_engine.transform([d.text for d in readable])
        old_labels, confidences = predict_labels(base_bundle, vectors)
    except Exception as exc:
        progress(f"⚠ Impossible de reprendre les catégories de l'ancien modèle : {exc}")
        return {}
    threshold = get_config().confidence_threshold
    return {
        doc.path: label
        for doc, label, confidence in zip(readable, old_labels, confidences)
        if confidence >= threshold
    }


def _build_bundle(
    documents: list[ExtractedDocument],
    engine_name: str,
    embedding_model: str | None,
    k_min: int,
    k_max: int,
    source_dirs: list[str],
    progress,
    previous_labels: dict[str, str] | None = None,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    min_silhouette: float | None = None,
    min_cluster_size: int | None = None,
) -> tuple[dict, list[ExtractedDocument], np.ndarray, dict[int, str]]:
    """Vectorise, regroupe et nomme les catégories. Cœur commun à `discover`
    (qui range aussi les fichiers) et `build_model` (qui ne fait que produire
    le modèle, pour l'onglet Entraînement de la GUI).

    `tfidf_max_features`, `tfidf_ngram_max`, `min_silhouette` et
    `min_cluster_size` permettent de passer outre les valeurs de la
    configuration pour CET entraînement précis (voir les préréglages de
    l'onglet Entraînement) sans changer les valeurs par défaut de
    l'application."""
    readable = [d for d in documents if not d.is_empty]
    unreadable_count = len(documents) - len(readable)
    if unreadable_count:
        progress(f"⚠ {unreadable_count} document(s) illisible(s) ignoré(s) pour le regroupement.")
    if len(readable) < 2:
        raise ValueError("Pas assez de documents lisibles pour former des catégories (minimum 2).")

    texts = [d.text for d in readable]
    engine = create_engine(
        engine_name,
        embedding_model=embedding_model,
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_max=tfidf_ngram_max,
    )

    if engine_name == "embeddings":
        progress(
            f"Calcul des embeddings pour {len(texts)} document(s) "
            "(le tout premier lancement peut être plus lent : téléchargement du modèle)..."
        )
    else:
        progress(f"Vectorisation TF-IDF de {len(texts)} document(s)...")
    vectors = engine.fit_transform(texts)

    # Seule limite au nombre de catégories : k_max (configurable, aucun
    # plafond caché en plus) et le nombre de documents lisibles lui-même —
    # on ne peut pas former plus de groupes distincts que de documents.
    effective_k_max = max(k_min, min(k_max, len(readable) - 1))
    progress(f"Recherche du nombre de catégories optimal ({k_min} à {effective_k_max})...")
    n_clusters = _best_k(vectors, k_min, effective_k_max, min_silhouette=min_silhouette, min_cluster_size=min_cluster_size)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(vectors)
    labels = km.labels_

    previous_names = [previous_labels.get(doc.path) for doc in readable] if previous_labels else None
    cluster_names, carried_over_ids, raw_names = _name_clusters(
        texts, labels, n_clusters, top_n=get_config().cluster_naming_top_words, previous_names=previous_names
    )

    progress(f"\n── {n_clusters} catégorie(s) détectée(s) ──")
    for cid, name in cluster_names.items():
        carried = " (reprise de l'ancien modèle)" if cid in carried_over_ids else ""
        progress(f"  {cid} → {name}{carried}")

    bundle = {
        "version": 1,
        "mode": "unsupervised",
        "engine_state": engine_to_state(engine),
        "km_model": km,
        "cluster_names": cluster_names,
        # Nom détecté par le modèle (mots-clés TF-IDF), jamais modifié par un
        # renommage manuel dans l'onglet Transformer les données — sert de
        # référence pour retrouver ce que le modèle a effectivement identifié.
        "original_cluster_names": raw_names,
        "source_dirs": source_dirs,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents_trained": len(readable),
    }
    return bundle, readable, labels, cluster_names


def discover(
    input_dir: str,
    output_dir: str,
    engine_name: str = ENGINE_TFIDF,
    embedding_model: str | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
    recursive: bool = False,
    move: bool = False,
    model_out: str | None = None,
    progress=print,
) -> dict:
    config = get_config()
    k_min = k_min if k_min is not None else config.cluster_k_min
    k_max = k_max if k_max is not None else config.cluster_k_max

    documents = extract_documents(input_dir, recursive=recursive, progress=progress)
    if not documents:
        raise ValueError(f"Aucun document pris en charge trouvé dans {input_dir}")

    unreadable = [d for d in documents if d.is_empty]
    bundle, readable, labels, cluster_names = _build_bundle(
        documents, engine_name, embedding_model, k_min, k_max, [input_dir], progress
    )

    results = {}
    for doc, label in zip(readable, labels):
        results[doc.path] = {"category": cluster_names[int(label)], "confidence": None}
    for doc in unreadable:
        results[doc.path] = {"category": unreadable_category(), "confidence": 0.0}

    progress("\n── Classement ──")
    for path, info in results.items():
        dest = dispatch_file(path, info["category"], output_dir, move=move)
        progress(f"  {path} → {dest}")

    manifest_path = output_dir.rstrip("/\\") + "_classification.json"
    write_json_atomic(
        {
            "categories": {str(k): v for k, v in cluster_names.items()},
            "files": {path: info["category"] for path, info in results.items()},
        },
        manifest_path,
    )

    if model_out:
        progress("\nEnregistrement du modèle...")
        model_store.save_bundle(bundle, model_out)
        progress(f"✓ Modèle réutilisable sauvegardé : {model_out}")

    return results


def build_model(
    input_dir: str,
    model_path: str,
    engine_name: str = ENGINE_TFIDF,
    embedding_model: str | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
    recursive: bool = False,
    base_model_path: str | None = None,
    sync_dataset: bool = True,
    progress=print,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    min_silhouette: float | None = None,
    min_cluster_size: int | None = None,
) -> dict:
    """Construit (ou améliore) un modèle à partir d'un dossier de documents
    non triés — utilisé par l'onglet Entraînement de la GUI, qui ne demande
    volontairement aucun tri manuel.

    Si `base_model_path` est fourni, les dossiers sources qui ont servi à le
    construire sont ré-extraits et combinés avec `input_dir` : le nouveau
    modèle est donc entraîné sur l'ancien jeu de documents ET les nouveaux.

    Si `sync_dataset` est vrai (par défaut), le dossier `dataset/` propre à
    ce modèle (à côté du .pkl — voir `model_store.model_dataset_dir`) est
    mis à jour avec une copie de chaque document, organisée par catégorie
    actuelle. Toujours en COPIE (jamais en déplacement) pour que les
    documents d'origine restent en place : une future amélioration du modèle
    (via `base_model_path`) doit pouvoir les retrouver au même endroit.
    """
    config = get_config()
    k_min = k_min if k_min is not None else config.cluster_k_min
    k_max = k_max if k_max is not None else config.cluster_k_max

    source_dirs = [input_dir]
    if base_model_path:
        base_bundle = model_store.load_bundle(base_model_path)
        base_dirs = base_bundle.get("source_dirs", [])
        for directory in base_dirs:
            if directory not in source_dirs:
                source_dirs.append(directory)
        progress(f"Amélioration du modèle {base_model_path} ({len(base_dirs)} dossier(s) source déjà connus).")

    all_documents: list[ExtractedDocument] = []
    seen_paths: set[str] = set()
    missing_dirs = []
    for directory in source_dirs:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
            continue
        for doc in extract_documents(directory, recursive=recursive, progress=progress):
            if doc.path not in seen_paths:
                all_documents.append(doc)
                seen_paths.add(doc.path)

    if missing_dirs:
        progress(f"⚠ Dossier(s) source introuvable(s), ignoré(s) : {', '.join(missing_dirs)}")
    if not all_documents:
        raise ValueError("Aucun document pris en charge trouvé.")

    previous_labels: dict[str, str] | None = None
    if base_model_path:
        progress("Reprise des catégories déjà connues (renommages/suppressions inclus)...")
        previous_labels = _infer_previous_labels(base_model_path, all_documents, progress)

    unreadable = [d for d in all_documents if d.is_empty]
    bundle, readable, labels, cluster_names = _build_bundle(
        all_documents, engine_name, embedding_model, k_min, k_max, source_dirs, progress,
        previous_labels=previous_labels,
        tfidf_max_features=tfidf_max_features, tfidf_ngram_max=tfidf_ngram_max,
        min_silhouette=min_silhouette, min_cluster_size=min_cluster_size,
    )

    if model_store.snapshot_model(model_path):
        progress("✓ État précédent archivé (revenir en arrière possible).")

    progress("\nEnregistrement du modèle...")
    model_store.save_bundle(bundle, model_path)
    progress(f"✓ Modèle enregistré : {model_path} ({len(readable)} document(s), {len(cluster_names)} catégorie(s))")

    if sync_dataset:
        _sync_dataset(model_path, readable, labels, unreadable, cluster_names, progress)

    return bundle


def improve_model(
    model_path: str,
    new_documents: list[ExtractedDocument],
    confirmed_labels: dict[str, str],
    k_min: int | None = None,
    k_max: int | None = None,
    sync_dataset: bool = True,
    progress=print,
) -> dict:
    """Améliore un modèle déjà entraîné avec un lot de documents dont la
    catégorie est déjà connue avec certitude (ex. corrections validées dans
    l'onglet Classification, case "Améliorer le modèle avec ces documents").

    `confirmed_labels` associe le chemin de chaque document de
    `new_documents` à sa catégorie confirmée : contrairement au reste du
    corpus (repris par vote majoritaire via l'ancien modèle, voir
    `_infer_previous_labels`), ces catégories priment toujours — ce sont des
    corrections humaines, pas des prédictions.

    Le résultat est réenregistré au même chemin (`model_path`). Ce lot de
    documents ne devient pas un `source_dir` permanent du modèle (contrairement
    à `build_model`) : ses catégories confirmées sont directement intégrées
    aux centres de regroupement, sans qu'il faille le re-scanner à chaque
    futur ré-entraînement — en revanche, si `sync_dataset` est vrai (par
    défaut), une copie de chaque document rejoint le dossier `dataset/`
    propre à ce modèle, donc reste consultable même après un redémarrage de
    l'application (voir `model_store.model_dataset_dir`).
    """
    config = get_config()
    k_min = k_min if k_min is not None else config.cluster_k_min
    k_max = k_max if k_max is not None else config.cluster_k_max

    base_bundle = model_store.load_bundle(model_path)
    source_dirs = base_bundle.get("source_dirs", [])

    historical_documents: list[ExtractedDocument] = []
    missing_dirs = []
    for directory in source_dirs:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
            continue
        historical_documents.extend(extract_documents(directory, recursive=False, progress=progress))
    if missing_dirs:
        progress(f"⚠ Dossier(s) source introuvable(s), ignoré(s) : {', '.join(missing_dirs)}")

    new_paths = {doc.path for doc in new_documents}
    all_documents = list(new_documents) + [doc for doc in historical_documents if doc.path not in new_paths]

    progress("Reprise des catégories déjà connues (renommages/suppressions inclus)...")
    previous_labels = _infer_previous_labels(model_path, all_documents, progress)
    previous_labels.update(confirmed_labels)  # les corrections confirmées priment toujours

    engine_state = base_bundle["engine_state"]
    bundle, readable, labels, cluster_names = _build_bundle(
        all_documents, engine_state["type"], engine_state.get("model_name"),
        k_min, k_max, source_dirs, progress,
        previous_labels=previous_labels,
    )

    if model_store.snapshot_model(model_path):
        progress("✓ État précédent archivé (revenir en arrière possible).")

    progress("\nEnregistrement du modèle...")
    model_store.save_bundle(bundle, model_path)
    progress(f"✓ Modèle amélioré : {model_path} ({len(readable)} document(s), {len(cluster_names)} catégorie(s))")

    if sync_dataset:
        unreadable = [doc for doc in all_documents if doc.is_empty]
        _sync_dataset(model_path, readable, labels, unreadable, cluster_names, progress)

    return bundle


def _sync_dataset(
    model_path: str,
    readable: list[ExtractedDocument],
    labels: np.ndarray,
    unreadable: list[ExtractedDocument],
    cluster_names: dict[int, str],
    progress,
) -> str:
    """Tient à jour storage/models/<nom>/dataset/<catégorie>/ : une copie de
    chaque document ayant contribué au modèle, organisée par sa catégorie
    ACTUELLE. Contrairement à un aperçu jetable, ce dossier est cumulatif et
    stable dans le temps — c'est lui que l'onglet Transformer les données
    consulte, et il reste accessible après un redémarrage de l'application
    puisqu'il se déduit uniquement du chemin du modèle, jamais d'un état
    volatile. Si un document change de catégorie d'un entraînement à
    l'autre, son ancienne copie est retirée pour éviter les doublons."""
    dataset_dir = model_store.model_dataset_dir(model_path)
    manifest_path = model_store.model_manifest_path(model_path)

    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    files_entry: dict[str, dict] = manifest.get("files", {})

    def sync_one(doc: ExtractedDocument, category: str) -> None:
        previous = files_entry.get(doc.path)
        if previous and previous.get("category") == category and os.path.exists(previous.get("dataset_path", "")):
            return  # déjà présent sous la bonne catégorie, rien à faire
        if previous and previous.get("dataset_path") and os.path.exists(previous["dataset_path"]):
            try:
                os.remove(previous["dataset_path"])
            except OSError:
                pass
        dest = dispatch_file(doc.path, category, dataset_dir, move=False)
        files_entry[doc.path] = {"category": category, "dataset_path": dest}

    for doc, label in zip(readable, labels):
        sync_one(doc, cluster_names[int(label)])
    for doc in unreadable:
        sync_one(doc, unreadable_category())

    # Un document déplacé vers une autre catégorie (ci-dessus) peut laisser
    # son ancien dossier de catégorie vide — le nettoyer plutôt que de le
    # laisser traîner indéfiniment sur le disque.
    if os.path.isdir(dataset_dir):
        for entry in os.listdir(dataset_dir):
            entry_path = os.path.join(dataset_dir, entry)
            if os.path.isdir(entry_path) and not os.listdir(entry_path):
                try:
                    os.rmdir(entry_path)
                except OSError:
                    pass

    manifest["categories"] = {str(k): v for k, v in cluster_names.items()}
    manifest["files"] = files_entry
    manifest["model_path"] = model_path
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json_atomic(manifest, manifest_path)
    progress(f"✓ Dataset du modèle à jour : {dataset_dir}")
    return dataset_dir
