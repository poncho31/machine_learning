"""Mode classification : applique un modèle déjà entraîné (`train` ou
`discover --model-out`) à de nouveaux documents, sans tout ré-entraîner.

Les prédictions peu fiables (probabilité/confiance sous `threshold`) sont
placées dans une catégorie "a_verifier" plutôt que d'être mal classées en
silence.
"""
from __future__ import annotations

import numpy as np

from . import model_store
from .config import get_config
from .extraction import ExtractedDocument, extract_documents
from .features import engine_from_state
from .utils import dispatch_file, write_json_atomic


def uncertain_category() -> str:
    return get_config().uncertain_category_name


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
    """Catégories connues du modèle, pour peupler une liste déroulante."""
    if bundle["mode"] == "supervised":
        return list(bundle["label_names"])
    return list(bundle["cluster_names"].values())


def load_model_for_prediction(model_path: str) -> tuple[dict, object]:
    bundle = model_store.load_bundle(model_path)
    engine = engine_from_state(bundle["engine_state"])
    return bundle, engine


def classify_documents(
    bundle: dict, engine, documents: list[ExtractedDocument], threshold: float | None = None
) -> dict[str, dict]:
    """Classe des documents déjà extraits. Réutilisé par la CLI, la GUI et
    l'automatisation, qui n'ont pas toutes la même façon d'obtenir la liste
    de documents à traiter (un dossier entier, une sélection, les nouveaux
    fichiers détectés par un job planifié...)."""
    if threshold is None:
        threshold = get_config().confidence_threshold
    readable = [d for d in documents if not d.is_empty]
    unreadable = [d for d in documents if d.is_empty]

    results: dict[str, dict] = {}

    if readable:
        vectors = engine.transform([d.text for d in readable])
        labels, confidences = predict_labels(bundle, vectors)
        for doc, label, confidence in zip(readable, labels, confidences):
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

    documents = extract_documents(input_dir, recursive=recursive)
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
