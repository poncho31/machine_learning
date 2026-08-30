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

Trois points d'entrée :
- `discover` : range aussi les fichiers dans des sous-dossiers (usage CLI).
- `build_model` : construit le modèle (utilisé par l'onglet Entraînement de
  la GUI), avec la possibilité de fusionner les documents d'un modèle
  existant pour l'améliorer (ré-entraînement complet, y compris le
  regroupement K-Means). Ne déplace jamais les documents d'origine, mais
  tient à jour une copie dans le dossier `dataset/` propre au modèle (voir
  `model_store.model_dataset_dir`) — c'est ce dossier, stable et déductible
  du seul chemin du modèle, que consulte la section "Catégories de ce
  modèle" de l'onglet Entraînement, y compris après un redémarrage de
  l'application.
- `improve_model` : lie des documents à une catégorie confirmée à la main
  depuis l'onglet Classification, SANS jamais relancer le regroupement
  K-Means ni toucher aux catégories et fichiers déjà présents (voir ce
  module, docstring de `improve_model`).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from . import model_store
from .classify import (
    confirmed_override,
    known_categories,
    load_model_for_prediction,
    predict_labels,
    uncertain_category,
    unreadable_category,
)
from .config import get_config
from .extraction import ExtractedDocument, extract_documents
from .formats import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS
from .features import (
    ENGINE_IMAGE,
    ENGINE_TFIDF,
    STOPWORDS,
    TOKEN_PATTERN,
    TfidfEngine,
    create_engine,
    engine_from_state,
    engine_to_state,
)
from .utils import (
    content_hash,
    detect_duplicate_pairs,
    dispatch_file,
    duplicate_removal_candidates,
    move_to_backup,
    write_json_atomic,
)


class _CentroidPredictor:
    """Enveloppe picklable exposant `.transform()` comme le ferait un
    `KMeans` (matrice des distances de chaque document à chaque centroïde),
    pour que `classify._predict_unsupervised` — qui appelle sans distinction
    `bundle["km_model"].transform(vectors)` — fonctionne à l'identique quel
    que soit l'algorithme de regroupement réellement utilisé.

    Nécessaire pour deux cas que `KMeans`/`MiniBatchKMeans` seuls ne
    couvrent pas : les algorithmes qui ne calculent pas eux-mêmes de
    centroïdes (`agglomerative`, `hdbscan` — les centroïdes sont alors
    recalculés à la main, comme la moyenne des vecteurs de chaque cluster),
    et le cas où une réduction de dimension (SVD) a été appliquée avant le
    regroupement : les vecteurs bruts du moteur (`engine.transform`, utilisés
    à la prédiction) doivent alors repasser par le MÊME SVD avant de pouvoir
    être comparés aux centroïdes, calculés eux dans l'espace réduit."""

    def __init__(self, centers: np.ndarray, svd: TruncatedSVD | None = None):
        self.cluster_centers_ = np.asarray(centers)
        self.svd = svd

    def transform(self, X):
        if self.svd is not None:
            X = self.svd.transform(X)
        return euclidean_distances(X, self.cluster_centers_)


def _make_estimator(algorithm: str, k: int):
    """Construit l'estimateur de regroupement pour un `k` donné. Seuls les
    algorithmes acceptant `n_clusters` passent par ici (voir `_best_k`) —
    "hdbscan", qui déduit son propre nombre de groupes des données, est géré
    séparément dans `_build_bundle`."""
    if algorithm == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=k)
    return KMeans(n_clusters=k, random_state=42, n_init="auto")


def _best_k(
    vectors: np.ndarray,
    k_min: int,
    k_max: int,
    algorithm: str = "kmeans",
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
        labels = _make_estimator(algorithm, k).fit_predict(vectors)
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


_keybert_model_cache: dict[str, object] = {}


def _get_keybert_model(model_name: str = "all-MiniLM-L6-v2"):
    """Charge (une seule fois, mis en cache pour le reste du process) le
    modèle KeyBERT utilisé pour nommer les catégories. Réutilise
    all-MiniLM-L6-v2 : déjà le modèle d'embeddings léger par défaut de
    l'application (voir features.DEFAULT_EMBEDDING_MODEL), pas un
    téléchargement supplémentaire."""
    if model_name not in _keybert_model_cache:
        from keybert import KeyBERT  # import différé : paquet optionnel

        _keybert_model_cache[model_name] = KeyBERT(model=model_name)
    return _keybert_model_cache[model_name]


def _keybert_name(text: str, boilerplate: set[str], top_n: int) -> str | None:
    """Nom d'un cluster via KeyBERT : sélectionne les mots/groupes de mots
    du texte les plus proches sémantiquement de l'ensemble du texte (via
    l'embedding du modèle), avec MMR pour diversifier plutôt que retenir des
    variantes du même mot-clé — généralement plus lisible qu'un assemblage
    des mots les mieux pondérés en TF-IDF. None si KeyBERT n'est pas
    disponible (paquet optionnel non installé) ou n'a rien pu extraire :
    l'appelant se replie alors sur le nommage TF-IDF existant."""
    text = text.strip()[:5000]  # au-delà, le modèle d'embeddings tronque de toute façon
    if not text:
        return None
    try:
        model = _get_keybert_model()
        keywords = model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words=STOPWORDS,
            use_mmr=True, diversity=0.6, top_n=top_n + 6,
        )
    except Exception:
        return None

    words = []
    for phrase, _score in keywords:
        cleaned = phrase.strip().lower()
        if not cleaned or cleaned in boilerplate:
            continue
        words.append(cleaned.replace(" ", "_"))
        if len(words) >= top_n:
            break
    return "_".join(words) if words else None


def _name_clusters(
    texts: list[str],
    labels: np.ndarray,
    n_clusters: int,
    top_n: int = 3,
    previous_names: list[str | None] | None = None,
    use_keybert: bool = False,
    extra_stopwords: frozenset[str] = frozenset(),
) -> tuple[dict[int, str], set[int], dict[int, str]]:
    """Nomme chaque cluster. Si `previous_names[i]` donne la catégorie que le
    document `i` avait dans un modèle précédent (renommages/suppressions
    compris — voir `_infer_previous_labels`), le nom majoritaire des
    documents d'un cluster est repris tel quel plutôt que régénéré : c'est
    ce qui permet à un ré-entraînement de ne pas perdre les catégories déjà
    renommées ou supprimées dans la section "Catégories de ce modèle" de
    l'onglet Entraînement.

    Si `use_keybert` est vrai, KeyBERT est essayé en premier pour chaque
    cluster (nom généralement plus naturel), avec repli automatique sur le
    nommage TF-IDF s'il échoue ou n'est pas disponible.

    Le nom "brut" basé sur les mots-clés TF-IDF est toujours calculé pour
    chaque cluster, même quand un nom repris est utilisé comme nom affiché —
    c'est ce nom brut, jamais écrasé par un renommage manuel, qu'affiche
    l'onglet Entraînement comme "nom détecté par le modèle"."""
    names: dict[int, str] = {}
    carried_over: set[int] = set()

    if previous_names is not None:
        # Deux clusters K-Means DISTINCTS peuvent chacun voter majoritairement
        # pour le MÊME nom précédent (ex. une catégorie autrefois large se
        # scinde naturellement en plusieurs sous-groupes au fil des
        # améliorations successives). Sans départage, les deux hériteraient
        # silencieusement du même nom, faisant "disparaître" des catégories
        # pourtant bien réelles et distinctes d'un ré-entraînement à l'autre
        # — mesuré : un modèle réel passé de 9 à 4 noms de catégorie affichés
        # en quelques améliorations successives, alors que le nombre RÉEL de
        # clusters restait constant (11) tout du long. Seul le cluster dont
        # le vote est le plus fort (le plus de documents de cette ancienne
        # catégorie) reprend le nom ; les autres retombent sur un nom frais
        # (mots-clés TF-IDF/KeyBERT ci-dessous), qui reflète leur propre
        # contenu plutôt que d'usurper le nom d'un autre cluster.
        candidates: list[tuple[int, str, int]] = []
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
                candidates.append((cluster_id, majority_name, count))

        claimed_names: set[str] = set()
        for cluster_id, majority_name, _count in sorted(candidates, key=lambda c: c[2], reverse=True):
            if majority_name in claimed_names:
                continue
            names[cluster_id] = majority_name
            carried_over.add(cluster_id)
            claimed_names.add(majority_name)

    cluster_texts = [""] * n_clusters
    for text, label in zip(texts, labels):
        cluster_texts[label] += " " + text

    # Écarte les mots présents dans la quasi-totalité des documents SOURCE
    # (adresse, nom de l'employeur, nom du destinataire...) : avec beaucoup
    # de petits clusters, leur IDF locale ne suffit plus à les distinguer
    # d'un vrai mot-clé de catégorie (voir _boilerplate_terms).
    boilerplate = _boilerplate_terms(texts)

    if any(t.strip() for t in cluster_texts):
        # `use_stemming=False` : un nom de catégorie doit toujours afficher de
        # vrais mots ("facture"), jamais une racine tronquée ("factur") — la
        # racinisation n'est utile que pour RAPPROCHER des vecteurs entre eux,
        # pas pour être lue par un humain.
        naming_engine = TfidfEngine(max_features=1000, use_stemming=False, extra_stopwords=extra_stopwords)
        matrix = naming_engine.fit_transform(cluster_texts)
        terms = naming_engine.vectorizer.get_feature_names_out()
    else:
        matrix, terms = None, None

    raw_names: dict[int, str] = {}
    for i in range(n_clusters):
        name = None
        if use_keybert:
            name = _keybert_name(cluster_texts[i], boilerplate, top_n)
        if name is None and matrix is not None:
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


def _parse_candidate_labels(raw: str) -> list[str]:
    """Convertit `image_cluster_labels` ("personnes, paysage, ...") en liste
    de libellés, dans l'ordre, casse d'origine préservée (contrairement à
    `_parse_extra_stopwords` : ces libellés SONT le nom de catégorie
    affiché, pas de normalisation nécessaire pour les comparer entre eux)."""
    return [label.strip() for label in raw.split(",") if label.strip()]


def _name_image_clusters(
    engine,
    vectors: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    candidate_labels: list[str],
    previous_names: list[str | None] | None = None,
) -> tuple[dict[int, str], set[int], dict[int, str]]:
    """Nomme chaque cluster D'IMAGES par le libellé candidat (config
    `image_cluster_labels`) le plus proche de son centroïde visuel —
    l'équivalent, pour le moteur "image", du nommage par mots-clés
    TF-IDF/KeyBERT de `_name_clusters` : un document image n'a pas de texte
    propre dont extraire des mots-clés, mais CLIP projette du texte dans le
    même espace vectoriel que les images (`engine.encode_texts`), ce qui
    permet cette comparaison "zero-shot" (aucun exemple d'entraînement
    nécessaire pour ces libellés).

    Même logique de reprise des noms précédents par vote majoritaire que
    `_name_clusters` (voir sa docstring), pour ne pas perdre un renommage
    manuel d'un entraînement à l'autre."""
    names: dict[int, str] = {}
    carried_over: set[int] = set()

    if previous_names is not None:
        candidates: list[tuple[int, str, int]] = []
        for cluster_id in range(n_clusters):
            votes = [
                previous_names[i] for i in range(len(labels)) if labels[i] == cluster_id and previous_names[i]
            ]
            if not votes:
                continue
            majority_name, count = Counter(votes).most_common(1)[0]
            if count / len(votes) >= 0.5:
                candidates.append((cluster_id, majority_name, count))
        claimed_names: set[str] = set()
        for cluster_id, majority_name, _count in sorted(candidates, key=lambda c: c[2], reverse=True):
            if majority_name in claimed_names:
                continue
            names[cluster_id] = majority_name
            carried_over.add(cluster_id)
            claimed_names.add(majority_name)

    if not candidate_labels:
        candidate_labels = ["image"]
    label_vectors = engine.encode_texts(candidate_labels)

    raw_names: dict[int, str] = {}
    for cluster_id in range(n_clusters):
        cluster_vectors = vectors[labels == cluster_id]
        if len(cluster_vectors) == 0:
            raw_name = candidate_labels[0]
        else:
            centroid = cluster_vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            similarities = label_vectors @ centroid
            raw_name = candidate_labels[int(np.argmax(similarities))]
        raw_names[cluster_id] = raw_name
        if cluster_id not in names:
            names[cluster_id] = raw_name

    return names, carried_over, raw_names


def detected_category_for_document(
    bundle: dict,
    engine,
    text: str,
    top_n: int | None = None,
    use_keybert: bool | None = None,
) -> str | None:
    """Nom "détecté" pour UN document confirmé à la main — utilisé pour
    `detected_category` dans le manifeste par `improve_model` et par
    `rename.move_files_to_category`/`rename.add_files_to_category`, pour que
    la section "Catégories détectées" affiche systématiquement un nom
    pertinent, jamais la mention générique "(confirmée manuellement)".

    Modèle SUPERVISÉ (CLI `train`, catégories définies par l'utilisateur via
    des sous-dossiers) : l'étiquette la plus proche prédite par le
    classifieur — seule notion de "détection" pertinente dans ce mode, sans
    mots-clés de cluster à extraire.

    Modèle NON supervisé (onglet Entraînement) : calculé EXACTEMENT comme le
    nommage d'un cluster K-Means (voir `_name_clusters` — mêmes mots-clés
    TF-IDF, ou KeyBERT si le modèle a été entraîné avec cette option), mais
    appliqué à ce seul document — surtout PAS un rapprochement avec les
    clusters DÉJÀ existants du modèle (ce qui a été tenté puis abandonné :
    un document dont le sujet est très éloigné du reste du corpus, ex. un
    mode d'emploi technique perdu au milieu de relevés bancaires, reste
    mathématiquement "le moins loin" d'un cluster existant qui n'a pourtant
    aucun rapport avec lui — mesuré : un document constaté "détecté" comme
    appartenant à un cluster bancaire alors que ses propres mots-clés
    n'avaient rien à voir). Un nom tiré des mots-clés dominants DU DOCUMENT
    LUI-MÊME est bien plus pertinent et honnête."""
    if not text.strip():
        return None
    if bundle["mode"] == "supervised":
        vector = engine.transform([text])
        probs = bundle["classifier"].predict_proba(vector)
        idx = int(probs.argmax(axis=1)[0])
        return bundle["label_names"][idx]

    config = get_config()
    resolved_top_n = top_n if top_n is not None else config.cluster_naming_top_words
    resolved_use_keybert = (
        use_keybert
        if use_keybert is not None
        else bundle.get("training_params", {}).get("use_keybert", config.cluster_use_keybert)
    )
    if resolved_use_keybert:
        name = _keybert_name(text, set(), resolved_top_n)
        if name is not None:
            return name
    keywords = _independent_keyword_lists([text], top_n=resolved_top_n)[0]
    return "_".join(keywords) if keywords else None


def _infer_previous_labels(base_model_path: str, documents: list[ExtractedDocument], progress) -> dict[str, str]:
    """Fait repasser des documents dans un modèle déjà entraîné pour
    connaître la catégorie qu'il leur attribuerait — y compris les
    renommages et les suppressions (fusionnées dans "autre") faits dans
    la section "Catégories de ce modèle" de l'onglet Entraînement, puisque
    `predict_labels` lit `cluster_names` tel qu'il est actuellement
    enregistré dans le modèle.

    Seules les prédictions suffisamment confiantes (même seuil que pour la
    classification) comptent comme "vote" : un document sur lequel l'ancien
    modèle est incertain ne doit pas imposer un ancien nom à un cluster qui
    correspond en réalité à un sujet nouveau."""
    try:
        base_bundle, base_engine = load_model_for_prediction(base_model_path)
    except Exception as exc:
        progress(f"⚠ Impossible de reprendre les catégories de l'ancien modèle : {exc}")
        return {}
    # Le moteur "image" vectorise le contenu visuel (voir `_build_bundle`) :
    # `.text`/`is_empty` (toujours vide sans OCR) n'a aucun rapport avec sa
    # lisibilité, et il faut lui donner des chemins de fichier, pas du texte.
    is_image_engine = base_bundle.get("engine_state", {}).get("type") == ENGINE_IMAGE
    if is_image_engine:
        readable = [
            d for d in documents
            if os.path.splitext(d.path)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(d.path)
        ]
    else:
        readable = [d for d in documents if not d.is_empty]
    if not readable:
        return {}
    try:
        vectorize_inputs = [d.path for d in readable] if is_image_engine else [d.text for d in readable]
        vectors = base_engine.transform(vectorize_inputs)
        old_labels, confidences = predict_labels(base_bundle, vectors)
    except Exception as exc:
        progress(f"⚠ Impossible de reprendre les catégories de l'ancien modèle : {exc}")
        return {}
    threshold = get_config().confidence_threshold
    previous_labels: dict[str, str] = {}
    for doc, label, confidence in zip(readable, old_labels, confidences):
        # Une correction déjà confirmée à la main prime toujours sur la
        # prédiction brute (voir `classify.confirmed_override`) : sans ce
        # contrôle, un document dont le cluster K-Means d'origine a
        # entre-temps disparu ou changé de nom perdrait sa catégorie
        # confirmée dès le ré-entraînement chaîné suivant. Sans objet pour le
        # moteur "image" : `confirmed_override` compare par empreinte de
        # texte, toujours vide (donc identique) pour une image sans OCR.
        override = None if is_image_engine else confirmed_override(base_bundle, doc.text)
        if override is not None:
            previous_labels[doc.path] = override
        elif confidence >= threshold:
            previous_labels[doc.path] = label
    return previous_labels


def _resolve_training_param(value, key: str, base_bundle: dict | None, config_default):
    """Un paramètre d'entraînement explicitement fourni (non None) est
    toujours prioritaire. Sinon, s'il s'agit d'une amélioration d'un modèle
    existant, reprend la valeur que CE modèle avait utilisée pour être
    construit (`training_params`, voir `_build_bundle`) plutôt que de
    silencieusement retomber sur la configuration globale actuelle, qui a pu
    changer depuis. En dernier recours seulement (modèle sans
    `training_params`, ex. entraîné avant l'ajout de ce champ, ou aucun
    modèle de base) : la configuration globale."""
    if value is not None:
        return value
    if base_bundle is not None:
        base_value = base_bundle.get("training_params", {}).get(key)
        if base_value is not None:
            return base_value
    return config_default


def _merge_confirmed_overrides(
    base_bundle: dict | None,
    all_documents: list[ExtractedDocument],
    new_confirmed_labels: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Une correction confirmée à la main (case "Améliorer le modèle avec ces
    documents" de l'onglet Classification) ne doit pas se limiter à influencer
    le nommage du cluster K-Means le plus proche (`_name_clusters`) : le
    clustering restant purement non supervisé, rien ne garantit qu'un
    document rejoigne le cluster portant le nom confirmé, ni qu'un futur
    passage de classification retrouve cette catégorie pour ce même document.

    On mémorise donc chaque confirmation par EMPREINTE DU CONTENU
    (`content_hash`, stable d'un export à l'autre, contrairement au chemin)
    dans `bundle["confirmed_overrides"]`, accumulée d'une amélioration à
    l'autre. Cette fonction combine les confirmations déjà connues du modèle
    de base avec les toutes nouvelles de cette passe, et retourne :
    - `hash_overrides` : à stocker tel quel dans le nouveau bundle ;
    - `path_overrides` : la même chose mais indexée par chemin de document
      pour ce passage précis (tous les documents de `all_documents` dont le
      contenu correspond à une confirmation connue), prête à l'emploi pour
      `_sync_dataset`/`_write_corpus_digest` afin que CHAQUE amélioration
      refile correctement les documents déjà confirmés, même ceux qui ne
      font pas partie du tout nouveau lot."""
    hash_overrides: dict[str, str] = dict((base_bundle or {}).get("confirmed_overrides", {}))
    new_confirmed_labels = new_confirmed_labels or {}
    text_by_path = {doc.path: doc.text for doc in all_documents}
    for path, category in new_confirmed_labels.items():
        text = text_by_path.get(path, "")
        if text.strip():
            hash_overrides[content_hash(text)] = category

    path_overrides: dict[str, str] = dict(new_confirmed_labels)
    for doc in all_documents:
        if doc.path in path_overrides or not doc.text.strip():
            continue
        matched = hash_overrides.get(content_hash(doc.text))
        if matched is not None:
            path_overrides[doc.path] = matched

    return hash_overrides, path_overrides


def _parse_extra_stopwords(raw: str) -> frozenset[str]:
    """Convertit `cluster_extra_stopwords` ("mot1, mot2, ...") en ensemble de
    mots normalisés. En minuscules : le tokenizer TF-IDF met lui-même tout en
    minuscules avant tokenisation (`lowercase=True`, y compris avec un
    tokenizer personnalisé — voir `TfidfVectorizer.build_preprocessor`), donc
    un mot ajouté ici avec une majuscule ne matcherait sinon jamais rien."""
    if not raw:
        return frozenset()
    return frozenset(word.strip().lower() for word in raw.split(",") if word.strip())


def _build_bundle(
    documents: list[ExtractedDocument],
    engine_name: str,
    embedding_model: str | None,
    k_min: int,
    k_max: int,
    source_dirs: list[str],
    progress,
    previous_labels: dict[str, str] | None = None,
    previous_raw_names: dict[str, str] | None = None,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    tfidf_use_stemming: bool | None = None,
    min_silhouette: float | None = None,
    min_cluster_size: int | None = None,
    extensions: tuple[str, ...] | None = None,
    use_keybert: bool | None = None,
    detect_duplicates: bool = False,
    cluster_algorithm: str | None = None,
    cluster_use_svd: bool | None = None,
    cluster_svd_components: int | None = None,
    cluster_extra_stopwords: str | None = None,
    image_cluster_labels: str | None = None,
) -> tuple[dict, list[ExtractedDocument], np.ndarray, dict[int, str], np.ndarray, object]:
    """Vectorise, regroupe et nomme les catégories. Cœur commun à `discover`
    (qui range aussi les fichiers) et `build_model` (qui ne fait que produire
    le modèle, pour l'onglet Entraînement de la GUI).

    `previous_raw_names` (nom affiché -> nom "détecté" qu'avait ce nom dans
    le modèle de base, voir `build_model`) gèle le nom "détecté" affiché
    d'une catégorie déjà établie : sans lui, ce nom serait recalculé à
    chaque entraînement à partir du contenu ACTUEL du cluster, qui peut
    légèrement varier avec l'ajout de nouveaux documents — une catégorie
    déjà validée par l'utilisateur ne doit plus voir ce nom changer.

    `tfidf_max_features`, `tfidf_ngram_max`, `min_silhouette`,
    `min_cluster_size` et `extensions` permettent de passer outre les valeurs
    de la configuration pour CET entraînement précis (voir les préréglages de
    l'onglet Entraînement) sans changer les valeurs par défaut de
    l'application. Les valeurs effectivement utilisées (résolues depuis la
    configuration si non précisées) sont enregistrées dans le bundle
    (`training_params`) : un nouvel entraînement avec `base_model_path` les
    reprend par défaut
    plutôt que de silencieusement retomber sur la configuration globale, qui
    a pu changer depuis — et une classification avec ce modèle ne recherche
    que les types de fichiers utilisés pour le construire. `use_keybert`
    active un nommage des catégories via KeyBERT (voir `_keybert_name`)
    plutôt que le simple assemblage de mots-clés TF-IDF.

    `detect_duplicates`, si vrai, écarte les documents quasi identiques
    AVANT le regroupement (pas seulement après coup, dans le résumé du
    corpus) : sans ça, un même document copié plusieurs fois se retrouvait
    tout de même vectorisé et regroupé plusieurs fois, alourdissant le
    modèle pour rien alors que le regroupement des VRAIS documents distincts
    n'en profite aucunement. Un seul exemplaire de chaque groupe de
    quasi-doublons est conservé (voir `utils.duplicate_removal_candidates`),
    en réutilisant les vecteurs déjà calculés — aucun calcul supplémentaire.

    `cluster_algorithm` sélectionne l'algorithme de regroupement
    ("kmeans"/"minibatch_kmeans"/"agglomerative"/"hdbscan" — voir la config
    `cluster_algorithm`). `cluster_use_svd`/`cluster_svd_components`
    réduisent la dimension des vecteurs TF-IDF avant regroupement (SVD/LSA,
    sans effet sur les embeddings). `cluster_extra_stopwords` exclut des mots
    supplémentaires du nommage ET de la vectorisation (voir
    `_parse_extra_stopwords`)."""
    config = get_config()
    resolved_tfidf_max_features = tfidf_max_features if tfidf_max_features is not None else config.tfidf_max_features
    resolved_tfidf_ngram_max = tfidf_ngram_max if tfidf_ngram_max is not None else config.tfidf_ngram_max
    resolved_tfidf_use_stemming = tfidf_use_stemming if tfidf_use_stemming is not None else config.tfidf_use_stemming
    resolved_min_silhouette = min_silhouette if min_silhouette is not None else config.cluster_min_silhouette
    resolved_min_cluster_size = min_cluster_size if min_cluster_size is not None else config.cluster_min_cluster_size
    resolved_extensions = tuple(extensions) if extensions is not None else SUPPORTED_EXTENSIONS
    resolved_use_keybert = use_keybert if use_keybert is not None else config.cluster_use_keybert
    resolved_cluster_algorithm = cluster_algorithm if cluster_algorithm is not None else config.cluster_algorithm
    resolved_cluster_use_svd = cluster_use_svd if cluster_use_svd is not None else config.cluster_use_svd
    resolved_cluster_svd_components = (
        cluster_svd_components if cluster_svd_components is not None else config.cluster_svd_components
    )
    resolved_cluster_extra_stopwords = (
        cluster_extra_stopwords if cluster_extra_stopwords is not None else config.cluster_extra_stopwords
    )
    extra_stopwords_set = _parse_extra_stopwords(resolved_cluster_extra_stopwords)
    resolved_image_cluster_labels = (
        image_cluster_labels if image_cluster_labels is not None else config.image_cluster_labels
    )

    if engine_name == ENGINE_IMAGE:
        # Le moteur "image" vectorise le CONTENU VISUEL du fichier (CLIP),
        # pas son texte : `.text`/`is_empty` (vide sans OCR, voir
        # formats._extract_image) ne dit rien de la lisibilité d'une image.
        # Seuls les vrais fichiers image comptent comme "lisibles" ici — un
        # autre format sélectionné par erreur avec ce moteur planterait au
        # chargement PIL plutôt que d'être filtré silencieusement.
        readable = [
            d for d in documents
            if os.path.splitext(d.path)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(d.path)
        ]
    else:
        readable = [d for d in documents if not d.is_empty]
    unreadable_count = len(documents) - len(readable)
    if unreadable_count:
        progress(f"⚠ {unreadable_count} document(s) illisible(s) ignoré(s) pour le regroupement.")
    if len(readable) < 2:
        raise ValueError("Pas assez de documents lisibles pour former des catégories (minimum 2).")

    texts = [d.text for d in readable]
    vectorize_inputs = [d.path for d in readable] if engine_name == ENGINE_IMAGE else texts
    engine = create_engine(
        engine_name,
        embedding_model=embedding_model,
        tfidf_max_features=resolved_tfidf_max_features,
        tfidf_ngram_max=resolved_tfidf_ngram_max,
        tfidf_use_stemming=resolved_tfidf_use_stemming,
        tfidf_extra_stopwords=extra_stopwords_set,
    )

    if engine_name == ENGINE_IMAGE:
        progress(
            f"Analyse visuelle (CLIP) de {len(vectorize_inputs)} image(s) "
            "(le tout premier lancement peut être plus lent : téléchargement du modèle)..."
        )
    elif engine_name == "embeddings":
        progress(
            f"Calcul des embeddings pour {len(texts)} document(s) "
            "(le tout premier lancement peut être plus lent : téléchargement du modèle)..."
        )
    else:
        progress(f"Vectorisation TF-IDF de {len(texts)} document(s)...")
    vectors = engine.fit_transform(vectorize_inputs)

    if detect_duplicates:
        config_for_duplicates = get_config()
        duplicate_pairs = detect_duplicate_pairs(
            [d.path for d in readable], [d.filename for d in readable], vectors,
            threshold=config_for_duplicates.cluster_duplicate_threshold,
            max_docs=config_for_duplicates.cluster_duplicate_max_docs,
        )
        to_exclude = set(duplicate_removal_candidates(duplicate_pairs))
        if to_exclude:
            keep_indices = [i for i, d in enumerate(readable) if d.path not in to_exclude]
            readable = [readable[i] for i in keep_indices]
            texts = [texts[i] for i in keep_indices]
            vectors = vectors[keep_indices]
            progress(
                f"⚠ {len(to_exclude)} document(s) en double écarté(s) avant l'entraînement "
                "(un seul exemplaire de chaque groupe de quasi-doublons est conservé)."
            )
            if len(readable) < 2:
                raise ValueError(
                    "Pas assez de documents lisibles et distincts pour former des catégories "
                    "(minimum 2) une fois les doublons écartés."
                )

    resolved_algorithm = resolved_cluster_algorithm
    if resolved_algorithm == "kmeans" and len(readable) > config.cluster_large_corpus_threshold:
        resolved_algorithm = "minibatch_kmeans"
        progress(
            f"Corpus de {len(readable)} documents > {config.cluster_large_corpus_threshold} : "
            "passage automatique à MiniBatchKMeans (plus rapide sur un gros corpus)."
        )

    # Réduction de dimension AVANT regroupement (pas avant le calcul des
    # doublons ci-dessus, qui doit rester sur l'espace TF-IDF complet). Sans
    # effet pour les embeddings, déjà denses et de dimension raisonnable
    # (voir la config `cluster_use_svd`). `vectors` (non réduit) reste utilisé
    # tel quel pour le nommage et le résumé du corpus.
    svd: TruncatedSVD | None = None
    cluster_vectors = vectors
    if engine_name == ENGINE_TFIDF and resolved_cluster_use_svd:
        n_components = min(resolved_cluster_svd_components, vectors.shape[1] - 1, len(readable) - 1)
        if n_components >= 2:
            progress(f"Réduction de dimension avant regroupement (SVD, {n_components} composantes)...")
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            cluster_vectors = svd.fit_transform(vectors)

    # Les catégories "hors normes" (bruit HDBSCAN) sont fusionnées dans cet
    # identifiant de cluster, nommé de force `config.other_category_name`
    # ci-dessous — jamais nommé par mots-clés comme un vrai cluster (voir la
    # config `cluster_algorithm`).
    other_cluster_id: int | None = None

    if resolved_algorithm == "hdbscan":
        progress("Regroupement HDBSCAN (nombre de catégories déduit des données, k_min/k_max ignorés)...")
        raw_labels = HDBSCAN(min_cluster_size=max(2, resolved_min_cluster_size)).fit_predict(cluster_vectors)
        noise_mask = raw_labels == -1
        unique_ids = sorted(set(raw_labels[~noise_mask].tolist()))
        remap = {old: new for new, old in enumerate(unique_ids)}
        n_clusters = len(unique_ids)
        if noise_mask.any():
            other_cluster_id = n_clusters
            n_clusters += 1
        if n_clusters == 0:
            # Tout le corpus est considéré comme "bruit" par HDBSCAN (corpus
            # trop petit ou trop homogène) : repli sur une seule catégorie
            # plutôt que sur un modèle sans aucune catégorie exploitable.
            labels = np.zeros(len(readable), dtype=int)
            n_clusters = 1
            other_cluster_id = None
        else:
            labels = np.array([remap[label] if label != -1 else other_cluster_id for label in raw_labels])
        centers = np.stack([cluster_vectors[labels == cid].mean(axis=0) for cid in range(n_clusters)])
        km = _CentroidPredictor(centers, svd=svd)
    else:
        # Seule limite au nombre de catégories : k_max (configurable, aucun
        # plafond caché en plus) et le nombre de documents lisibles lui-même —
        # on ne peut pas former plus de groupes distincts que de documents.
        effective_k_max = max(k_min, min(k_max, len(readable) - 1))
        progress(f"Recherche du nombre de catégories optimal ({k_min} à {effective_k_max})...")
        n_clusters = _best_k(
            cluster_vectors, k_min, effective_k_max, algorithm=resolved_algorithm,
            min_silhouette=resolved_min_silhouette, min_cluster_size=resolved_min_cluster_size,
        )
        estimator = _make_estimator(resolved_algorithm, n_clusters).fit(cluster_vectors)
        labels = estimator.labels_
        if resolved_algorithm in ("kmeans", "minibatch_kmeans") and svd is None:
            # Cas le plus courant : l'estimateur lui-même expose déjà
            # `.transform()` sur le même espace vectoriel que celui utilisé à
            # la prédiction (voir `classify._predict_unsupervised`), pas
            # besoin de l'envelopper dans `_CentroidPredictor`.
            km = estimator
        else:
            centers = np.stack([cluster_vectors[labels == cid].mean(axis=0) for cid in range(n_clusters)])
            km = _CentroidPredictor(centers, svd=svd)

    if config.cluster_report_extra_metrics and n_clusters >= 2 and len(set(labels.tolist())) >= 2:
        try:
            progress(
                "Qualité du regroupement — silhouette : {:.3f} | "
                "Davies-Bouldin : {:.3f} (plus bas = mieux) | "
                "Calinski-Harabasz : {:.1f} (plus haut = mieux)".format(
                    silhouette_score(cluster_vectors, labels),
                    davies_bouldin_score(cluster_vectors, labels),
                    calinski_harabasz_score(cluster_vectors, labels),
                )
            )
        except ValueError:
            pass

    previous_names = [previous_labels.get(doc.path) for doc in readable] if previous_labels else None
    if engine_name == ENGINE_IMAGE:
        progress("Nommage des catégories par comparaison visuelle (CLIP, zero-shot)...")
        candidate_labels = _parse_candidate_labels(resolved_image_cluster_labels)
        cluster_names, carried_over_ids, raw_names = _name_image_clusters(
            engine, vectors, labels, n_clusters, candidate_labels, previous_names=previous_names,
        )
    else:
        if resolved_use_keybert:
            progress("Nommage des catégories via KeyBERT...")
        cluster_names, carried_over_ids, raw_names = _name_clusters(
            texts, labels, n_clusters, top_n=get_config().cluster_naming_top_words, previous_names=previous_names,
            use_keybert=resolved_use_keybert, extra_stopwords=extra_stopwords_set,
        )
    if other_cluster_id is not None:
        cluster_names[other_cluster_id] = config.other_category_name
        raw_names[other_cluster_id] = config.other_category_name

    # Un cluster qui reprend un nom affiché déjà établi (`carried_over_ids`
    # — vote majoritaire ci-dessus, voir `_name_clusters`/`_name_image_clusters`)
    # doit AUSSI reprendre le nom "détecté" qu'avait cette catégorie, plutôt
    # que d'en recalculer un nouveau à partir du contenu actuel du cluster
    # (qui a pu légèrement changer avec l'ajout de nouveaux documents) : une
    # catégorie déjà établie/validée par l'utilisateur ne doit plus jamais
    # voir son nom "détecté par le modèle" changer sous ses yeux, seules les
    # catégories réellement NOUVELLES (aucun nom précédent à reprendre) sont
    # nommées à neuf.
    if previous_raw_names:
        for cid in carried_over_ids:
            frozen = previous_raw_names.get(cluster_names[cid])
            if frozen:
                raw_names[cid] = frozen

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
        # renommage manuel dans l'onglet Entraînement — sert de référence
        # pour retrouver ce que le modèle a effectivement identifié.
        "original_cluster_names": raw_names,
        "source_dirs": source_dirs,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents_trained": len(readable),
        # Paramètres effectivement utilisés pour CET entraînement (résolus
        # depuis la configuration si non précisés) : repris par défaut par un
        # nouvel entraînement via `base_model_path` (voir `build_model`), et
        # par la classification pour ne scanner que les types de fichiers
        # utilisés à la construction.
        "training_params": {
            "k_min": k_min,
            "k_max": k_max,
            "min_silhouette": resolved_min_silhouette,
            "min_cluster_size": resolved_min_cluster_size,
            "tfidf_max_features": resolved_tfidf_max_features,
            "tfidf_ngram_max": resolved_tfidf_ngram_max,
            "tfidf_use_stemming": resolved_tfidf_use_stemming,
            "extensions": resolved_extensions,
            "use_keybert": resolved_use_keybert,
            # Algorithme EFFECTIVEMENT utilisé (après le passage automatique
            # kmeans -> minibatch_kmeans au-delà de cluster_large_corpus_threshold,
            # voir plus haut), pas nécessairement celui demandé.
            "cluster_algorithm": resolved_algorithm,
            "cluster_use_svd": resolved_cluster_use_svd,
            "cluster_svd_components": resolved_cluster_svd_components,
            "cluster_extra_stopwords": resolved_cluster_extra_stopwords,
            "image_cluster_labels": resolved_image_cluster_labels,
        },
    }
    return bundle, readable, labels, cluster_names, vectors, engine


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
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    tfidf_use_stemming: bool | None = None,
    use_keybert: bool | None = None,
    detect_duplicates: bool | None = None,
    cluster_algorithm: str | None = None,
    cluster_use_svd: bool | None = None,
    cluster_svd_components: int | None = None,
    cluster_extra_stopwords: str | None = None,
    image_cluster_labels: str | None = None,
) -> dict:
    config = get_config()
    k_min = k_min if k_min is not None else config.cluster_k_min
    k_max = k_max if k_max is not None else config.cluster_k_max
    detect_duplicates = detect_duplicates if detect_duplicates is not None else config.cluster_detect_duplicates

    documents = extract_documents(input_dir, recursive=recursive, progress=progress)
    if not documents:
        raise ValueError(f"Aucun document pris en charge trouvé dans {input_dir}")

    bundle, readable, labels, cluster_names, _vectors, _engine = _build_bundle(
        documents, engine_name, embedding_model, k_min, k_max, [input_dir], progress,
        tfidf_max_features=tfidf_max_features, tfidf_ngram_max=tfidf_ngram_max,
        tfidf_use_stemming=tfidf_use_stemming, use_keybert=use_keybert, detect_duplicates=detect_duplicates,
        cluster_algorithm=cluster_algorithm, cluster_use_svd=cluster_use_svd,
        cluster_svd_components=cluster_svd_components, cluster_extra_stopwords=cluster_extra_stopwords,
        image_cluster_labels=image_cluster_labels,
    )
    # Calculé APRÈS _build_bundle plutôt qu'avec `is_empty` avant : pour le
    # moteur "image", la lisibilité ne dépend pas de `.text` (voir
    # `_build_bundle`) — seule la liste `readable` qu'il retourne fait foi.
    readable_paths = {d.path for d in readable}
    unreadable = [d for d in documents if d.path not in readable_paths]

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
    recursive: bool | None = None,
    base_model_path: str | None = None,
    sync_dataset: bool = True,
    progress=print,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    tfidf_use_stemming: bool | None = None,
    min_silhouette: float | None = None,
    min_cluster_size: int | None = None,
    extensions: tuple[str, ...] | None = None,
    use_keybert: bool | None = None,
    detect_duplicates: bool | None = None,
    cluster_algorithm: str | None = None,
    cluster_use_svd: bool | None = None,
    cluster_svd_components: int | None = None,
    cluster_extra_stopwords: str | None = None,
    image_cluster_labels: str | None = None,
) -> dict:
    """Construit (ou améliore) un modèle à partir d'un dossier de documents
    non triés — utilisé par l'onglet Entraînement de la GUI, qui ne demande
    volontairement aucun tri manuel.

    `extensions` restreint les fichiers pris en compte lors du scan de
    `input_dir` (voir l'onglet Entraînement, sélection des types de fichiers) :
    un seul type, plusieurs, ou tous les formats pris en charge (défaut).

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
    base_bundle = model_store.load_bundle(base_model_path) if base_model_path else None
    k_min = _resolve_training_param(k_min, "k_min", base_bundle, config.cluster_k_min)
    k_max = _resolve_training_param(k_max, "k_max", base_bundle, config.cluster_k_max)
    min_silhouette = _resolve_training_param(min_silhouette, "min_silhouette", base_bundle, config.cluster_min_silhouette)
    min_cluster_size = _resolve_training_param(
        min_cluster_size, "min_cluster_size", base_bundle, config.cluster_min_cluster_size
    )
    tfidf_max_features = _resolve_training_param(
        tfidf_max_features, "tfidf_max_features", base_bundle, config.tfidf_max_features
    )
    tfidf_ngram_max = _resolve_training_param(tfidf_ngram_max, "tfidf_ngram_max", base_bundle, config.tfidf_ngram_max)
    tfidf_use_stemming = _resolve_training_param(
        tfidf_use_stemming, "tfidf_use_stemming", base_bundle, config.tfidf_use_stemming
    )
    extensions = tuple(_resolve_training_param(extensions, "extensions", base_bundle, SUPPORTED_EXTENSIONS))
    use_keybert = _resolve_training_param(use_keybert, "use_keybert", base_bundle, config.cluster_use_keybert)
    detect_duplicates = detect_duplicates if detect_duplicates is not None else config.cluster_detect_duplicates
    cluster_algorithm = _resolve_training_param(
        cluster_algorithm, "cluster_algorithm", base_bundle, config.cluster_algorithm
    )
    cluster_use_svd = _resolve_training_param(cluster_use_svd, "cluster_use_svd", base_bundle, config.cluster_use_svd)
    cluster_svd_components = _resolve_training_param(
        cluster_svd_components, "cluster_svd_components", base_bundle, config.cluster_svd_components
    )
    cluster_extra_stopwords = _resolve_training_param(
        cluster_extra_stopwords, "cluster_extra_stopwords", base_bundle, config.cluster_extra_stopwords
    )
    image_cluster_labels = _resolve_training_param(
        image_cluster_labels, "image_cluster_labels", base_bundle, config.image_cluster_labels
    )
    # Par défaut True (comme la case "Inclure les sous-dossiers" de l'onglet
    # Entraînement) : les dossiers sources d'un modèle chaîné via
    # `base_model_path` sont très souvent le `dataset/` d'un AUTRE modèle,
    # organisé en sous-dossiers par catégorie (`dataset/<catégorie>/*.pdf`) —
    # un scan non récursif de ce genre de dossier ne trouve tout simplement
    # RIEN.
    recursive = _resolve_training_param(recursive, "recursive", base_bundle, True)

    source_dirs = [input_dir]
    if base_model_path:
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
        for doc in extract_documents(directory, recursive=recursive, progress=progress, extensions=extensions):
            if doc.path not in seen_paths:
                all_documents.append(doc)
                seen_paths.add(doc.path)

    if missing_dirs:
        progress(f"⚠ Dossier(s) source introuvable(s), ignoré(s) : {', '.join(missing_dirs)}")
        # Filet de sécurité, seulement quand au moins un dossier source est devenu introuvable
        # (ex. un ancien modèle de chaînage déplacé ou supprimé) : le
        # dataset/ du modèle de base — toujours présent tant que ce modèle
        # lui-même existe — est alors rescanné en plus, pour ne pas perdre
        # les documents qu'il contenait déjà. Volontairement PAS rescanné
        # quand tous les dossiers sources sont valides, sinon chaque document
        # s'y retrouverait compté deux fois (l'original ET sa copie dans
        # dataset/), gonflant artificiellement le corpus.
        if base_model_path:
            base_dataset_dir = model_store.model_dataset_dir(base_model_path)
            if os.path.isdir(base_dataset_dir):
                for doc in extract_documents(base_dataset_dir, recursive=True, progress=progress, extensions=extensions):
                    if doc.path not in seen_paths:
                        all_documents.append(doc)
                        seen_paths.add(doc.path)

    if not all_documents:
        raise ValueError("Aucun document pris en charge trouvé.")

    previous_labels: dict[str, str] | None = None
    previous_raw_names: dict[str, str] | None = None
    if base_model_path:
        progress("Reprise des catégories déjà connues (renommages/suppressions inclus)...")
        previous_labels = _infer_previous_labels(base_model_path, all_documents, progress)
        # Nom "détecté" qu'avait chaque catégorie dans le modèle de base —
        # gèle ce nom pour toute catégorie reprise ci-dessous (voir
        # `_build_bundle`, paramètre `previous_raw_names`) plutôt que de le
        # laisser dériver à chaque entraînement.
        previous_raw_names = {
            name: base_bundle.get("original_cluster_names", {}).get(cid)
            for cid, name in base_bundle.get("cluster_names", {}).items()
        }

    bundle, readable, labels, cluster_names, vectors, engine = _build_bundle(
        all_documents, engine_name, embedding_model, k_min, k_max, source_dirs, progress,
        previous_labels=previous_labels, previous_raw_names=previous_raw_names,
        tfidf_max_features=tfidf_max_features, tfidf_ngram_max=tfidf_ngram_max,
        tfidf_use_stemming=tfidf_use_stemming,
        min_silhouette=min_silhouette, min_cluster_size=min_cluster_size,
        extensions=extensions, use_keybert=use_keybert, detect_duplicates=detect_duplicates,
        cluster_algorithm=cluster_algorithm, cluster_use_svd=cluster_use_svd,
        cluster_svd_components=cluster_svd_components, cluster_extra_stopwords=cluster_extra_stopwords,
        image_cluster_labels=image_cluster_labels,
    )
    # Calculé APRÈS _build_bundle plutôt qu'avec `is_empty` avant : pour le
    # moteur "image", la lisibilité ne dépend pas de `.text` (voir
    # `_build_bundle`) — seule la liste `readable` qu'il retourne fait foi.
    # Sans ce recalcul, une image correctement regroupée se retrouvait AUSSI
    # listée comme "illisible" ci-dessous, et `_sync_dataset` l'écrasait dans
    # la catégorie fourre-tout illisible juste après l'y avoir correctement
    # classée.
    readable_paths = {d.path for d in readable}
    unreadable = [d for d in all_documents if d.path not in readable_paths]
    hash_overrides, path_overrides = _merge_confirmed_overrides(base_bundle, all_documents)
    bundle["confirmed_overrides"] = hash_overrides
    bundle["training_params"]["recursive"] = recursive

    if model_store.snapshot_model(model_path):
        progress("✓ État précédent archivé (revenir en arrière possible).")

    progress("\nEnregistrement du modèle...")
    model_store.save_bundle(bundle, model_path)
    progress(f"✓ Modèle enregistré : {model_path} ({len(readable)} document(s), {len(cluster_names)} catégorie(s))")

    if sync_dataset:
        _sync_dataset(
            model_path, readable, labels, unreadable, cluster_names, progress,
            category_overrides=path_overrides, original_names=bundle.get("original_cluster_names", {}),
        )

    _write_corpus_digest(
        model_path, engine_name, engine, vectors, readable, labels, cluster_names, detect_duplicates, progress,
        category_overrides=path_overrides,
    )

    return bundle


def _add_confirmed_documents_to_dataset(
    model_path: str,
    new_documents: list[ExtractedDocument],
    confirmed_labels: dict[str, str],
    progress,
    detected_labels: dict[str, str] | None = None,
) -> int:
    """Ajoute chaque document confirmé à la main (onglet Classification,
    "Améliorer le modèle avec ces documents") au dossier `dataset/` du modèle
    et à son manifeste — voir `improve_model`, qui appelle cette fonction en
    tout premier et n'en attend rien d'autre (aucun regroupement K-Means,
    aucune revectorisation du reste du corpus derrière). Une copie de
    fichier ne peut pas échouer pour les mêmes raisons qu'un recalcul de
    regroupement (score de silhouette insuffisant, erreur du moteur
    d'analyse...) : le fichier catégorisé à la main est donc garanti de
    rejoindre le modèle. Retourne le nombre de documents effectivement
    ajoutés/mis à jour.

    `detected_labels` (chemin -> catégorie) associe, quand elle est connue,
    la catégorie que le modèle avait PROPOSÉE pour ce document avant sa
    correction manuelle (voir `gui.ClassifyTab._validate`,
    `row.predicted_category`) — enregistrée dans le manifeste comme
    `detected_category`, exactement comme le ferait un cluster K-Means (voir
    `_sync_dataset`). Sans elle, `gui._populate_categories` n'a aucune
    catégorie "détectée" à afficher pour un document confirmé et retombe sur
    la mention générique "(confirmée manuellement)"."""
    dataset_dir = model_store.model_dataset_dir(model_path)
    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})
    detected_labels = detected_labels or {}

    documents_by_path = {doc.path: doc for doc in new_documents}
    added = 0
    for path, category in confirmed_labels.items():
        if documents_by_path.get(path) is None or not os.path.exists(path):
            continue
        detected = detected_labels.get(path)
        previous = files_entry.get(path)
        if (
            previous
            and previous.get("category") == category
            and (detected is None or previous.get("detected_category") == detected)
            and os.path.exists(previous.get("dataset_path", ""))
        ):
            continue
        if previous and previous.get("dataset_path") and os.path.exists(previous["dataset_path"]):
            try:
                os.remove(previous["dataset_path"])
            except OSError:
                pass
        dest = dispatch_file(path, category, dataset_dir, move=False)
        entry = {"category": category, "dataset_path": dest}
        if detected is not None:
            entry["detected_category"] = detected
        files_entry[path] = entry
        added += 1

    if added:
        manifest["files"] = files_entry
        manifest["model_path"] = model_path
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        model_store.save_manifest(manifest, model_path)
        progress(f"✓ {added} document(s) confirmé(s) ajouté(s) au dataset du modèle.")
    return added


def improve_model(
    model_path: str,
    new_documents: list[ExtractedDocument],
    confirmed_labels: dict[str, str],
    progress=print,
) -> dict:
    """Lie un lot de documents à une catégorie CONFIRMÉE À LA MAIN (onglet
    Classification, case "Améliorer le modèle avec ces documents" ; route API
    `/improve`) — jamais une simple prédiction.

    Contrairement à un ré-entraînement complet (`build_model` avec
    `base_model_path`, onglet Entraînement), cette opération ne relance
    JAMAIS le regroupement K-Means et ne revectorise jamais le reste du
    corpus : le modèle existant (clusters, noms de catégorie, autres
    documents) n'est ni détruit ni restructuré. Elle se contente, pour
    chaque document confirmé, de :
    1. calculer son nom "détecté" à partir de SES PROPRES mots-clés
       dominants (`detected_category_for_document`, mêmes mots-clés TF-IDF
       ou KeyBERT qu'un cluster K-Means — voir `_name_clusters`), TOUJOURS —
       indépendamment de tout seuil de confiance et SANS chercher à le
       rapprocher des clusters déjà existants du modèle (qui peut être
       arbitraire pour un document dont le sujet est trop éloigné du reste
       du corpus) ;
    2. copier le document dans `dataset/<catégorie>/` et inscrire ce nom
       comme `detected_category` dans le manifeste du modèle
       (`_add_confirmed_documents_to_dataset`) — la catégorie existe alors
       immédiatement dans la section "Catégories détectées", AVEC un nom
       détecté renseigné, même si elle est entièrement nouvelle et ne
       correspond à aucun cluster (voir `classify.known_categories`,
       `gui._populate_categories` : la mention générique "(confirmée
       manuellement)" ne doit donc plus jamais apparaître) ;
    3. enregistrer l'EMPREINTE DE SON CONTENU (`content_hash`) dans
       `bundle["confirmed_overrides"]`, consultée en priorité à chaque
       classification future (voir `classify.classify_documents`) : ce même
       document, où qu'il réapparaisse, retrouve toujours sa catégorie
       confirmée avec une confiance de 100 %, sans dépendre du clustering.

    `confirmed_labels` associe le chemin de chaque document de
    `new_documents` à sa catégorie retenue. Seules les entrées dont la
    catégorie n'est PAS un simple repli ("à vérifier" / "non catégorisé",
    voir `config.uncertain_category_name`/`unreadable_category_name`) sont
    prises en compte : un repli ne constitue pas une correction, seulement
    l'absence de décision — l'incohérence serait d'apprendre au modèle
    qu'un document appartient à "à vérifier". Ce filtre est appliqué ICI,
    pas seulement côté GUI, pour protéger aussi les appelants de la route API
    `/improve`.

    Volontairement SANS condition sur `bundle["mode"]` : contrairement à
    `rename.rename_categories`/`move_files_to_category`/`add_files_to_category`
    (réservées aux modèles non supervisés — renommer/fusionner des clusters
    n'a pas de sens pour un modèle supervisé aux catégories déjà fixées par
    l'utilisateur), lier un document à une catégorie par empreinte de
    contenu est un mécanisme générique qui fonctionne pour N'IMPORTE QUEL
    modèle (voir `classify.confirmed_override`, utilisé sans condition de
    mode). Un modèle entraîné via la CLI (`train`, mode "supervised") peut
    donc aussi recevoir des `confirmed_overrides` ; `classify.known_categories`
    en tient compte pour les deux modes, précisément pour que ce cas reste
    cohérent partout où les catégories d'un modèle sont énumérées.
    """
    placeholder_categories = {uncertain_category(), unreadable_category()}
    confirmed_labels = {
        path: category for path, category in confirmed_labels.items() if category not in placeholder_categories
    }

    bundle = model_store.load_bundle(model_path)
    if not confirmed_labels:
        progress(
            "Aucune correction exploitable (catégorie \"à vérifier\"/\"non catégorisé\" ignorée) : "
            "le modèle n'a pas été modifié."
        )
        return bundle

    if model_store.snapshot_model(model_path):
        progress("✓ État précédent archivé (revenir en arrière possible).")

    documents_by_path = {doc.path: doc for doc in new_documents}
    engine = engine_from_state(bundle["engine_state"])
    detected_labels: dict[str, str] = {}
    for path in confirmed_labels:
        doc = documents_by_path.get(path)
        if doc is None or not doc.text.strip():
            continue
        detected = detected_category_for_document(bundle, engine, doc.text)
        if detected is not None:
            detected_labels[path] = detected

    added = _add_confirmed_documents_to_dataset(
        model_path, new_documents, confirmed_labels, progress, detected_labels=detected_labels
    )

    hash_overrides = dict(bundle.get("confirmed_overrides", {}))
    digest_batch: list[tuple[ExtractedDocument, str]] = []
    for path, category in confirmed_labels.items():
        doc = documents_by_path.get(path)
        if doc is None or not doc.text.strip():
            continue
        hash_overrides[content_hash(doc.text)] = category
        digest_batch.append((doc, category))

    bundle["confirmed_overrides"] = hash_overrides
    model_store.save_bundle(bundle, model_path)

    if digest_batch:
        _write_confirmed_digest_entries(model_path, digest_batch, progress)

    progress(
        f"✓ Modèle amélioré : {model_path} ({added} document(s) lié(s) à une catégorie, "
        f"{len(known_categories(bundle))} catégorie(s) au total). "
        "Les catégories et fichiers déjà présents n'ont pas été touchés."
    )
    return bundle


def _sync_dataset(
    model_path: str,
    readable: list[ExtractedDocument],
    labels: np.ndarray,
    unreadable: list[ExtractedDocument],
    cluster_names: dict[int, str],
    progress,
    category_overrides: dict[str, str] | None = None,
    original_names: dict[int, str] | None = None,
) -> str:
    """Tient à jour storage/models/<nom>/dataset/<catégorie>/ : une copie de
    chaque document ayant contribué au modèle, organisée par sa catégorie
    ACTUELLE. Contrairement à un aperçu jetable, ce dossier est cumulatif et
    stable dans le temps — c'est lui que la section "Catégories de ce
    modèle" de l'onglet Entraînement consulte, et il reste accessible après
    un redémarrage de l'application
    puisqu'il se déduit uniquement du chemin du modèle, jamais d'un état
    volatile. Si un document change de catégorie d'un entraînement à
    l'autre, son ancienne copie est retirée pour éviter les doublons.

    `category_overrides` (chemin de document -> catégorie) prend le pas sur
    la catégorie du cluster KMeans auquel le document a été rattaché —
    nécessaire, lors d'un ré-entraînement complet chaîné (`build_model` avec
    `base_model_path`), pour les documents dont la catégorie avait déjà été
    confirmée à la main lors d'une amélioration précédente (voir
    `improve_model`, `_merge_confirmed_overrides`) : le clustering K-Means,
    purement non supervisé, ne garantit PAS qu'un document rejoigne le
    cluster portant le nom confirmé — cette confirmation ne pesait
    auparavant que comme un vote parmi d'autres dans le nommage du cluster
    (voir `_name_clusters`), pas comme une affectation garantie, ce qui
    pouvait faire disparaître une correction pourtant validée à la main de
    son dossier de catégorie.

    `original_names` (cluster_id -> nom brut détecté par le moteur, TF-IDF ou
    KeyBERT selon ce qui était activé — voir `bundle["original_cluster_names"]`)
    est enregistré par document dans le manifest (`detected_category`), à
    côté de sa catégorie ACTUELLE (`category`, potentiellement corrigée à la
    main) : les deux restent distincts et consultables même après une
    correction manuelle, pour toujours pouvoir comparer ce que le modèle a
    proposé et ce qui a été confirmé."""
    dataset_dir = model_store.model_dataset_dir(model_path)
    category_overrides = category_overrides or {}
    original_names = original_names or {}

    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})

    def sync_one(doc: ExtractedDocument, category: str, detected: str | None = None) -> None:
        category = category_overrides.get(doc.path, category)
        previous = files_entry.get(doc.path)
        if (
            previous and previous.get("category") == category
            and previous.get("detected_category") == detected
            and os.path.exists(previous.get("dataset_path", ""))
        ):
            return  # déjà présent sous la bonne catégorie, rien à faire
        if previous and previous.get("dataset_path") and os.path.exists(previous["dataset_path"]):
            try:
                os.remove(previous["dataset_path"])
            except OSError:
                pass
        dest = dispatch_file(doc.path, category, dataset_dir, move=False)
        entry = {"category": category, "dataset_path": dest}
        if detected is not None:
            entry["detected_category"] = detected
        files_entry[doc.path] = entry

    for doc, label in zip(readable, labels):
        cluster_id = int(label)
        sync_one(doc, cluster_names[cluster_id], detected=original_names.get(cluster_id))
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

    # Le manifest ne doit garder qu'UNE entrée par fichier RÉELLEMENT présent
    # dans dataset/ aujourd'hui. Sans ce nettoyage, chaque document accumule
    # indéfiniment une entrée par chemin sous lequel il a jamais été
    # découvert (dossier source d'origine, copie dataset/ d'une génération
    # de modèle précédente rescannée en filet de sécurité — voir
    # `build_model`...) : le .json gonfle avec des entrées orphelines
    # (fichier disparu ailleurs) ou redondantes (plusieurs chemins pointant
    # vers la même copie actuelle), sans rien ajouter d'utile.
    seen_dataset_paths: set[str] = set()
    pruned_files_entry: dict[str, dict] = {}
    for source_path, entry in files_entry.items():
        dataset_path = entry.get("dataset_path") if isinstance(entry, dict) else None
        if not dataset_path or not os.path.exists(dataset_path):
            continue
        normalized = os.path.normpath(dataset_path)
        if normalized in seen_dataset_paths:
            continue
        seen_dataset_paths.add(normalized)
        pruned_files_entry[source_path] = entry
    files_entry = pruned_files_entry

    manifest["categories"] = {str(k): v for k, v in cluster_names.items()}
    manifest["files"] = files_entry
    manifest["model_path"] = model_path
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_store.save_manifest(manifest, model_path)
    progress(f"✓ Dataset du modèle à jour : {dataset_dir}")
    return dataset_dir


_DIGEST_EXCERPT_LENGTH = 500
_DIGEST_KEYWORDS_PER_DOC = 8


def _keywords_from_matrix(matrix, terms, top_n: int) -> list[list[str]]:
    keyword_lists: list[list[str]] = []
    for row in matrix:
        top_idx = row.argsort()[-top_n:][::-1]
        keyword_lists.append([terms[j] for j in top_idx if row[j] > 0])
    return keyword_lists


def _independent_keyword_lists(texts: list[str], top_n: int = _DIGEST_KEYWORDS_PER_DOC) -> list[list[str]]:
    """Mots-clés par document via une passe TF-IDF légère et indépendante sur
    CE lot de textes uniquement (voir `_document_keyword_lists`, dont c'est
    le repli en embeddings) — utilisé aussi par `improve_model`, qui ne
    dispose que des documents tout juste confirmés à la main et ne
    revectorise jamais le reste du corpus."""
    # `use_stemming=False` : ces mots-clés sont affichés tels quels (nom
    # "détecté" d'un document dans l'onglet Entraînement/Classification),
    # jamais une racine tronquée ("factur") — voir `_name_clusters`, même
    # choix pour le nommage des clusters.
    keyword_engine = TfidfEngine(max_features=2000, use_stemming=False)
    try:
        matrix = keyword_engine.fit_transform(texts)
    except ValueError:
        # "empty vocabulary" : après filtrage des mots vides et des jetons
        # trop courts/numériques (TOKEN_PATTERN), il ne reste plus un seul
        # terme exploitable — un texte très court (ex. un e-mail bref) le
        # déclenche facilement. Ce n'est pas une erreur, juste rien à en
        # tirer comme mots-clés ; ne doit surtout pas remonter jusqu'à
        # planter l'appelant (voir `detected_category_for_document`, appelé
        # en direct dans le fil de prédiction de la GUI).
        return [[] for _ in texts]
    terms = keyword_engine.vectorizer.get_feature_names_out()
    return _keywords_from_matrix(matrix, terms, top_n)


def _document_keyword_lists(
    engine_name: str, engine, vectors: np.ndarray, texts: list[str], top_n: int = _DIGEST_KEYWORDS_PER_DOC
) -> list[list[str]]:
    """Mots-clés par document pour le résumé du corpus (voir
    `_write_corpus_digest`). En TF-IDF, réutilise directement les vecteurs
    déjà calculés pour le regroupement (gratuit, aucun calcul
    supplémentaire) : chaque poids de terme est déjà le score TF-IDF du
    document. En embeddings, les vecteurs ne sont pas interprétables par
    terme — une passe TF-IDF légère et indépendante, sur le même corpus,
    sert uniquement à extraire des mots-clés lisibles."""
    if engine_name == ENGINE_TFIDF:
        terms = engine.vectorizer.get_feature_names_out()
        return _keywords_from_matrix(vectors, terms, top_n)
    return _independent_keyword_lists(texts, top_n)


def _excerpt(text: str, length: int = _DIGEST_EXCERPT_LENGTH) -> str:
    """Extrait lisible du début d'un document, coupé sur un espace pour ne
    pas trancher un mot en plein milieu."""
    text = text.strip()
    if len(text) <= length:
        return text
    truncated = text[:length]
    cut = truncated.rfind(" ")
    if cut > length // 2:  # évite une coupe absurdement courte si peu d'espaces
        truncated = truncated[:cut]
    return truncated + "…"


def _write_confirmed_digest_entries(
    model_path: str,
    confirmed_documents: list[tuple[ExtractedDocument, str]],
    progress,
) -> None:
    """Complète le manifeste (`<nom>.json`) avec mots-clés/extrait/nombre de
    caractères pour les documents tout juste confirmés à la main (voir
    `improve_model`) — le pendant, réduit à ce seul lot, de
    `_write_corpus_digest` (qui traite tout le corpus lors d'un
    entraînement). Ne touche à AUCUNE autre entrée du manifeste : les
    documents déjà connus du modèle gardent leur résumé existant tel quel."""
    keyword_lists = _independent_keyword_lists([doc.text for doc, _ in confirmed_documents])

    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})

    for (doc, category), keywords in zip(confirmed_documents, keyword_lists):
        entry = files_entry.setdefault(doc.path, {})
        entry["category"] = category
        entry["char_count"] = len(doc.text)
        entry["keywords"] = keywords
        entry["excerpt"] = _excerpt(doc.text)

    manifest["files"] = files_entry
    manifest["model_path"] = model_path
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_store.save_manifest(manifest, model_path)
    progress(f"✓ Résumé enregistré pour {len(confirmed_documents)} document(s) confirmé(s).")


def _detect_duplicates(
    readable: list[ExtractedDocument],
    vectors: np.ndarray,
    threshold: float | None = None,
    max_docs: int | None = None,
) -> list[dict]:
    """Détecte les paires de documents quasi identiques (voir
    `utils.detect_duplicate_pairs`, la logique partagée avec l'onglet
    Classification). Ajoute `file_a`/`file_b` (alias de `filename_a`/
    `filename_b`, conservés pour compatibilité avec le format déjà utilisé
    dans le résumé du corpus)."""
    config = get_config()
    threshold = threshold if threshold is not None else config.cluster_duplicate_threshold
    max_docs = max_docs if max_docs is not None else config.cluster_duplicate_max_docs

    pairs = detect_duplicate_pairs(
        [d.path for d in readable], [d.filename for d in readable], vectors,
        threshold=threshold, max_docs=max_docs,
    )
    for pair in pairs:
        pair["file_a"] = pair["filename_a"]
        pair["file_b"] = pair["filename_b"]
    return pairs


def _move_dataset_duplicates_to_backup(dataset_paths: list[str]) -> list[str]:
    """Déplace des copies en double du `dataset/` d'un modèle vers UN SEUL
    dossier `dataset/_backup/`, qui reprend en dessous de lui la même
    structure par catégorie (`dataset/_backup/<catégorie>/<fichier>`) —
    plutôt qu'un `_backup` séparé dans chaque sous-dossier de catégorie
    (`dataset/<catégorie>/_backup/`), plus difficile à retrouver et à vider
    d'un coup quand les doublons touchent plusieurs catégories à la fois.

    Chaque chemin de `dataset_paths` est de la forme
    `<dataset_dir>/<catégorie>/<fichier>` (voir `_sync_dataset`) : la
    catégorie et la racine du dataset s'en déduisent directement, sans
    connaître le chemin du modèle."""
    by_category_root: dict[tuple[str, str], list[str]] = {}
    for path in dataset_paths:
        category_dir = os.path.dirname(path)
        category = os.path.basename(category_dir)
        dataset_root = os.path.dirname(category_dir)
        by_category_root.setdefault((dataset_root, category), []).append(path)

    moved: list[str] = []
    for (dataset_root, category), group in by_category_root.items():
        backup_dir = os.path.join(dataset_root, "_backup", category)
        moved.extend(move_to_backup(group, backup_dir))
    return moved


def delete_training_duplicates(manifest_path: str, progress=print) -> list[str]:
    """Déplace vers un dossier `_backup` les copies jugées "en trop" parmi
    les doublons détectés lors du dernier entraînement (voir
    `_write_corpus_digest`, qui les enregistre dans le manifest `<nom>.json`
    du modèle — plus de fichier `_digest.json` séparé), en gardant toujours
    un exemplaire de chaque groupe de quasi-doublons. Jamais une suppression
    définitive, et surtout : **ne touche jamais aux documents d'origine
    (dossier source)** — seule leur COPIE dans `dataset/` (le dossier de
    destination propre à ce modèle, géré par l'application) est déplacée,
    vers `dataset/_backup/` (un seul dossier de secours à la racine du
    dataset, avec la même structure par catégorie en dessous — voir
    `_move_dataset_duplicates_to_backup`). Un doublon dont la copie dans
    `dataset/` est introuvable (ex. `sync_dataset=False` lors de
    l'entraînement) est ignoré plutôt que de risquer de toucher le fichier
    source.

    Utilisé par l'onglet Entraînement après affichage des doublons trouvés,
    quand l'utilisateur confirme vouloir les retirer."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    duplicates = manifest.get("duplicates", [])
    if not duplicates:
        return []

    to_remove_sources = duplicate_removal_candidates(duplicates)
    dataset_path_by_source: dict[str, str] = {}
    for pair in duplicates:
        if pair.get("dataset_path_a"):
            dataset_path_by_source[pair["path_a"]] = pair["dataset_path_a"]
        if pair.get("dataset_path_b"):
            dataset_path_by_source[pair["path_b"]] = pair["dataset_path_b"]

    to_remove_dataset = [dataset_path_by_source[p] for p in to_remove_sources if p in dataset_path_by_source]
    skipped = len(to_remove_sources) - len(to_remove_dataset)
    if skipped:
        progress(
            f"⚠ {skipped} doublon(s) sans copie dans dataset/ (sync_dataset désactivé ?), ignoré(s) — "
            "les documents source ne sont jamais déplacés automatiquement."
        )

    moved = _move_dataset_duplicates_to_backup(to_remove_dataset)
    progress(
        f"✓ {len(moved)} copie(s) en double déplacée(s) vers dataset/_backup/ "
        "(les documents d'origine ne sont jamais touchés)."
    )
    return moved


def _write_corpus_digest(
    model_path: str,
    engine_name: str,
    engine,
    vectors: np.ndarray,
    readable: list[ExtractedDocument],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    detect_duplicates: bool,
    progress,
    category_overrides: dict[str, str] | None = None,
) -> str:
    """Enrichit le manifest du modèle (`<nom>.json`, déjà mis à jour par
    `_sync_dataset` juste avant) avec un résumé condensé par document — PAS
    le texte intégral, mais mots-clés et court extrait — directement dans
    l'entrée de chaque fichier, plutôt que dans un fichier séparé
    `_digest.json` : un seul fichier fait référence pour un même modèle,
    jamais deux qui pourraient se désynchroniser. Ce résumé, beaucoup plus
    petit que le texte brut, est pensé pour servir de corpus à un futur
    modèle d'IA local (ex. proposer un nom de catégorie, résumer un fichier)
    sans avoir à ré-extraire les documents d'origine à chaque fois. Détecte
    aussi, si demandé, les paires de documents quasi identiques."""
    category_overrides = category_overrides or {}
    texts = [d.text for d in readable]
    keyword_lists = _document_keyword_lists(engine_name, engine, vectors, texts)

    manifest_path = model_store.model_manifest_path(model_path)
    manifest = model_store.load_manifest(model_path)
    files_entry: dict[str, dict] = manifest.get("files", {})

    for doc, label, keywords in zip(readable, labels, keyword_lists):
        entry = files_entry.setdefault(doc.path, {})
        entry["category"] = category_overrides.get(doc.path, cluster_names[int(label)])
        entry["char_count"] = len(doc.text)
        entry["keywords"] = keywords
        entry["excerpt"] = _excerpt(doc.text)

    duplicates = _detect_duplicates(readable, vectors) if detect_duplicates else []
    if duplicates:
        # La détection compare les documents SOURCE, mais une éventuelle
        # suppression ne doit jamais toucher les fichiers d'origine — voir
        # `delete_training_duplicates` : seule leur COPIE dans dataset/
        # (déjà à jour dans `files_entry` ci-dessus, puisque _sync_dataset
        # vient de tourner) doit pouvoir être visée.
        source_to_dataset_path = {
            src: entry.get("dataset_path")
            for src, entry in files_entry.items()
            if isinstance(entry, dict) and entry.get("dataset_path")
        }
        for pair in duplicates:
            pair["dataset_path_a"] = source_to_dataset_path.get(pair["path_a"])
            pair["dataset_path_b"] = source_to_dataset_path.get(pair["path_b"])

    manifest["files"] = files_entry
    manifest["duplicates"] = duplicates
    manifest["model_path"] = model_path
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_store.save_manifest(manifest, model_path)
    progress(f"✓ Résumé du corpus enregistré ({len(readable)} document(s)) : {manifest_path}")
    if detect_duplicates:
        if duplicates:
            progress(f"⚠ {len(duplicates)} paire(s) de documents quasi identiques détectée(s) :")
            for pair in duplicates[:10]:
                progress(f"    {pair['file_a']} ≈ {pair['file_b']} ({pair['similarity']:.0%})")
            if len(duplicates) > 10:
                progress(f"    ... et {len(duplicates) - 10} autre(s) — détail complet dans {manifest_path}")
        else:
            progress("✓ Aucun document en double détecté.")

    # Nettoyage : un ancien `_digest.json` séparé d'avant l'unification des
    # deux fichiers n'est plus lu par rien — le supprimer plutôt que de le
    # laisser traîner, périmé, à côté du manifest désormais complet.
    legacy_digest_path = model_store.model_digest_path(model_path)
    if os.path.exists(legacy_digest_path):
        try:
            os.remove(legacy_digest_path)
        except OSError:
            pass

    return manifest_path
