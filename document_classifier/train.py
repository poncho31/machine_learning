"""Mode entraînement : apprend un classifieur à partir de PDF déjà triés par
l'utilisateur dans des sous-dossiers (chaque sous-dossier = une catégorie).

C'est le cœur de l'usage visé : chaque utilisateur range un échantillon de ses
propres documents dans des dossiers représentant ses catégories, puis entraîne
un modèle qui lui est propre, réutilisable ensuite hors-ligne avec `classify`.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from . import model_store
from .extraction import extract_documents
from .features import ENGINE_TFIDF, create_engine, engine_to_state


def train(
    labeled_dir: str,
    model_path: str,
    engine_name: str = ENGINE_TFIDF,
    embedding_model: str | None = None,
    progress=print,
) -> dict:
    if not os.path.isdir(labeled_dir):
        raise NotADirectoryError(f"Dossier introuvable : {labeled_dir}")

    categories = sorted(
        d for d in os.listdir(labeled_dir) if os.path.isdir(os.path.join(labeled_dir, d))
    )
    if len(categories) < 2:
        raise ValueError(
            f"Il faut au moins 2 dossiers de catégories sous {labeled_dir} pour entraîner un modèle "
            "(un sous-dossier par catégorie, contenant les PDF déjà triés de cette catégorie)."
        )

    texts: list[str] = []
    labels: list[str] = []
    skipped: list[str] = []
    for category in categories:
        category_dir = os.path.join(labeled_dir, category)
        for doc in extract_documents(category_dir):
            if doc.is_empty:
                skipped.append(doc.path)
                continue
            texts.append(doc.text)
            labels.append(category)

    counts = Counter(labels)
    empty_categories = [c for c in categories if counts.get(c, 0) == 0]
    for category in empty_categories:
        progress(f"⚠ Catégorie {category!r} ignorée : aucun document lisible dedans.")
    categories = [c for c in categories if c not in empty_categories]

    if len(categories) < 2:
        raise ValueError(
            "Il reste moins de 2 catégories avec des documents lisibles : impossible d'entraîner un modèle."
        )

    underfilled = [c for c in categories if counts.get(c, 0) < 2]
    if underfilled:
        progress(
            f"⚠ Catégorie(s) avec un seul document lisible (pas d'évaluation possible) : {', '.join(underfilled)}"
        )

    if skipped:
        progress(f"⚠ {len(skipped)} PDF ignoré(s) au total (texte illisible) :")
        for path in skipped:
            progress(f"    {path}")

    kwargs = {"embedding_model": embedding_model} if embedding_model else {}
    engine = create_engine(engine_name, **kwargs)
    vectors = engine.fit_transform(texts)

    label_names = sorted(set(labels))
    label_index = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_index[label] for label in labels])

    report = None
    min_class_count = min(counts.values())
    if min_class_count >= 2 and len(texts) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            vectors, y, test_size=0.2, stratify=y, random_state=42
        )
        eval_clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        report = classification_report(
            y_test, eval_clf.predict(X_test), target_names=label_names, zero_division=0
        )
        progress("\n── Évaluation (20% des documents mis de côté) ──")
        progress(report)
    else:
        progress(
            "\n⚠ Pas assez de documents pour une évaluation fiable "
            "(le modèle est tout de même entraîné sur toutes les données disponibles)."
        )

    classifier = LogisticRegression(max_iter=1000).fit(vectors, y)

    bundle = {
        "version": 1,
        "mode": "supervised",
        "engine_state": engine_to_state(engine),
        "classifier": classifier,
        "label_names": label_names,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents_trained": len(texts),
        "categories_count": dict(counts),
    }
    model_store.save_bundle(bundle, model_path)
    progress(f"\n✓ Modèle entraîné sur {len(texts)} documents ({len(categories)} catégories) : {model_path}")

    return {"bundle": bundle, "report": report, "skipped": skipped}
