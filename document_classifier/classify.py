"""Mode classification : applique un modèle déjà entraîné (`train` ou
`discover --model-out`) à de nouveaux documents, sans tout ré-entraîner.

Les prédictions peu fiables (probabilité/confiance sous `threshold`) sont
placées dans une catégorie "a_verifier" plutôt que d'être mal classées en
silence.
"""
from __future__ import annotations

import os

import numpy as np

from . import model_store
from .config import get_config
from .extraction import ExtractedDocument, extract_documents
from .features import ENGINE_IMAGE, engine_from_state
from .formats import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS
from .utils import content_hash, dispatch_file, write_json_atomic


def uncertain_category() -> str:
    return get_config().uncertain_category_name


def model_extensions(bundle: dict) -> tuple[str, ...]:
    """Types de fichiers utilisés pour entraîner ce modèle (voir
    `discover._build_bundle`, `training_params`), pour qu'une classification
    ou une automatisation ne recherche que ces types-là plutôt que tous les
    formats pris en charge — un modèle entraîné uniquement sur des `.pdf` ne
    doit pas se mettre à considérer des `.docx` lors de son utilisation.
    Retombe sur tous les formats pris en charge si l'information n'est pas
    disponible (modèle entraîné avant l'ajout de ce champ, ou modèle
    supervisé via la CLI)."""
    extensions = bundle.get("training_params", {}).get("extensions")
    return tuple(extensions) if extensions else SUPPORTED_EXTENSIONS


def unreadable_category() -> str:
    return get_config().unreadable_category_name


def _predict_supervised(bundle: dict, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
    probs = bundle["classifier"].predict_proba(vectors)
    pred_idx = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    label_names = bundle["label_names"]
    labels = [label_names[i] for i in pred_idx]
    return labels, confidences


def _predict_unsupervised(bundle: dict, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
    km = bundle["km_model"]
    distances = km.transform(vectors)
    if distances.shape[1] < 2:
        # Un seul cluster (voir discover._best_k : aucune séparation assez
        # nette n'a été trouvée dans les données d'entraînement) — il n'y a
        # pas de second cluster dont mesurer l'écart, donc pas d'ambiguïté
        # possible : confiance maximale pour tous les documents.
        nearest = np.zeros(len(vectors), dtype=int)
        confidences = np.ones(len(vectors))
    else:
        order = np.argsort(distances, axis=1)
        nearest = order[:, 0]
        d_sorted = np.take_along_axis(distances, order, axis=1)
        second = np.where(d_sorted[:, 1] == 0, 1e-9, d_sorted[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            confidences = np.clip(1 - (d_sorted[:, 0] / second), 0, 1)
    cluster_names = bundle["cluster_names"]
    labels = [cluster_names[int(c)] for c in nearest]
    return labels, confidences


def predict_labels(bundle: dict, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Prédit (catégorie, confiance) pour des vecteurs déjà calculés, quel que
    soit le type de modèle (supervisé ou non). Réutilisé par la CLI et par
    l'interface graphique."""
    if bundle["mode"] == "supervised":
        return _predict_supervised(bundle, vectors)
    return _predict_unsupervised(bundle, vectors)


def known_categories(bundle: dict) -> list[str]:
    """Catégories connues du modèle, pour peupler une liste déroulante.

    `confirmed_overrides` (voir `discover.improve_model`) est fusionné dans
    les deux modes : cette fonction ne restreint elle-même jamais quel type
    de modèle peut recevoir une correction confirmée — une catégorie
    confirmée sur un modèle supervisé (CLI `train`) classe déjà correctement
    les documents identiques via `confirmed_override`/`classify_documents`,
    quel que soit le mode ; l'ignorer ici la rendrait invisible dans les
    menus déroulants tout en fonctionnant silencieusement à la classification."""
    if bundle["mode"] == "supervised":
        names = list(bundle["label_names"])
    else:
        names = list(bundle["cluster_names"].values())
    for extra in bundle.get("confirmed_overrides", {}).values():
        if extra not in names:
            names.append(extra)
    return names


def load_model_for_prediction(model_path: str) -> tuple[dict, object]:
    bundle = model_store.load_bundle(model_path)
    engine = engine_from_state(bundle["engine_state"])
    return bundle, engine


def confirmed_override(bundle: dict, text: str) -> str | None:
    """Catégorie confirmée à la main (onglet Classification, "Améliorer le
    modèle avec ces documents" ; route API `/improve`) pour un contenu
    IDENTIQUE à `text`, si connue — voir `discover.improve_model`, qui
    l'enregistre par empreinte de contenu (`content_hash`) dans
    `bundle["confirmed_overrides"]`, précisément parce que le clustering
    K-Means, purement non supervisé, ne garantit pas de reclasser un document
    au même endroit d'une fois sur l'autre.

    Centralise cette vérification pour que TOUT appelant qui prédit une
    catégorie (`classify_documents` ci-dessous — CLI, automatisation ; mais
    aussi la GUI, `gui.ClassifyTab._predict_paths`, quand elle prédit en
    direct pour peupler la liste "Documents à classer") reconnaisse un
    document déjà confirmé de la même façon. Un appelant qui interrogerait
    `predict_labels` directement sans passer par ici manquerait cette
    correction et proposerait à nouveau la prédiction brute du clustering."""
    if not text.strip():
        return None
    return bundle.get("confirmed_overrides", {}).get(content_hash(text))


def classify_documents(
    bundle: dict, engine, documents: list[ExtractedDocument], threshold: float | None = None
) -> dict[str, dict]:
    """Classe des documents déjà extraits. Réutilisé par la CLI, la GUI et
    l'automatisation, qui n'ont pas toutes la même façon d'obtenir la liste
    de documents à traiter (un dossier entier, une sélection, les nouveaux
    fichiers détectés par un job planifié...)."""
    if threshold is None:
        threshold = get_config().confidence_threshold
    # Le moteur "image" (voir discover._build_bundle) vectorise le contenu
    # visuel du fichier : `.text`/`is_empty` (toujours vide sans OCR) ne dit
    # rien de sa lisibilité, et il lui faut des chemins de fichier, pas du
    # texte.
    is_image_engine = bundle.get("engine_state", {}).get("type") == ENGINE_IMAGE
    if is_image_engine:
        def _is_readable(d: ExtractedDocument) -> bool:
            return os.path.splitext(d.path)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(d.path)
    else:
        def _is_readable(d: ExtractedDocument) -> bool:
            return not d.is_empty
    readable = [d for d in documents if _is_readable(d)]
    unreadable = [d for d in documents if not _is_readable(d)]

    results: dict[str, dict] = {}

    if readable:
        vectorize_inputs = [d.path for d in readable] if is_image_engine else [d.text for d in readable]
        vectors = engine.transform(vectorize_inputs)
        labels, confidences = predict_labels(bundle, vectors)
        for doc, label, confidence in zip(readable, labels, confidences):
            # Sans objet pour "image" : `confirmed_override` compare par
            # empreinte de TEXTE, toujours vide (donc identique) pour une
            # image sans OCR — voir discover._infer_previous_labels, même
            # limite documentée.
            override = None if is_image_engine else confirmed_override(bundle, doc.text)
            if override is not None:
                category, confidence = override, 1.0
            else:
                category = label if confidence >= threshold else uncertain_category()
            results[doc.path] = {"category": category, "confidence": float(confidence)}

    for doc in unreadable:
        results[doc.path] = {"category": unreadable_category(), "confidence": 0.0}

    return results


def classify(
    input_dir: str,
    model_path: str,
    output_dir: str,
    threshold: float | None = None,
    recursive: bool = False,
    move: bool = False,
    progress=print,
) -> dict:
    bundle, engine = load_model_for_prediction(model_path)

    documents = extract_documents(input_dir, recursive=recursive, extensions=model_extensions(bundle))
    if not documents:
        raise ValueError(f"Aucun document pris en charge trouvé dans {input_dir}")

    results = classify_documents(bundle, engine, documents, threshold=threshold)

    progress("\n── Classification ──")
    for path, info in results.items():
        dest = dispatch_file(path, info["category"], output_dir, move=move)
        conf = info["confidence"]
        conf_display = f"{conf:.0%}" if conf is not None else "n/a"
        progress(f"  {path} → {dest}  (confiance {conf_display})")

    manifest_path = output_dir.rstrip("/\\") + "_classification.json"
    write_json_atomic(
        {path: info for path, info in results.items()},
        manifest_path,
    )

    return results
