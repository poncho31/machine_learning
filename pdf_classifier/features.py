"""Moteurs de vectorisation du texte.

- TfidfEngine (par défaut) : aucune dépendance lourde, aucun téléchargement,
  fonctionne hors-ligne dès l'installation.
- EmbeddingEngine (optionnel) : meilleure qualité sémantique via
  sentence-transformers, nécessite un téléchargement de modèle (~90 Mo) au
  tout premier lancement puis fonctionne hors-ligne (modèle mis en cache).
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import get_config
from .stopwords_fr import FRENCH_STOPWORDS

ENGINE_TFIDF = "tfidf"
ENGINE_EMBEDDINGS = "embeddings"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Catalogue de modèles sentence-transformers pertinents pour de la
# catégorisation de documents, du plus léger (rapide, peu précis) au plus
# lourd (lent, plus précis) — évite d'avoir à connaître un nom de modèle par
# cœur. Tous se téléchargent une seule fois puis fonctionnent hors-ligne.
#
# ── TF-IDF ou embeddings : lequel choisir ? ──
# Mesuré sur un vrai jeu de test (16 fiches de paie très similaires, dont 1
# document en réalité différent — un récapitulatif fiscal annuel glissé dans
# le même dossier) : quand les documents à catégoriser sont très gabarités
# (même mise en page, vocabulaire quasi identique d'un document à l'autre —
# factures d'un même fournisseur, fiches de paie, relevés d'un même
# organisme...), AUCUN moteur ne trouve de vraie structure multi-catégorie,
# et c'est normal : il n'y en a pas. Dans ce cas TF-IDF est le meilleur choix
# par défaut — gratuit (millisecondes), aucun téléchargement, et pas moins
# pertinent que les embeddings puisqu'il n'y a rien à gagner en compréhension
# sémantique. Les embeddings ne creusent l'écart que sur des corpus
# RÉELLEMENT hétérogènes en texte libre (contrats de nature variée, emails,
# rapports) où deux documents de la même catégorie peuvent se ressembler
# sans partager le même vocabulaire exact.
#
# Contre-intuitif mais mesuré : sur des documents très gabarités où la seule
# variation d'un exemplaire à l'autre est numérique (dates, montants) plutôt
# que lexicale, un modèle multilingue ne catégorise pas mieux en français
# qu'un modèle anglais — la compréhension du sens n'aide pas quand ce qui
# différencie les documents n'est de toute façon pas sémantique. Un modèle
# multilingue reste préférable pour du texte libre réellement rédigé en
# français (contrats, courriers, rapports), mais n'attendez pas de gain
# automatique juste parce que vos documents sont en français.
EMBEDDING_MODEL_CATALOG = [
    (
        "all-MiniLM-L6-v2",
        "Léger (~90 Mo) — le plus rapide. Anglais principalement. Recommandé par défaut pour "
        "démarrer, y compris sur des documents très gabarités (factures, fiches de paie) où "
        "aucun modèle n'a d'avantage sémantique à faire valoir.",
    ),
    (
        "all-MiniLM-L12-v2",
        "Léger (~120 Mo) — un peu plus précis que L6, reste rapide. Anglais principalement. "
        "Bon compromis si TF-IDF peine à séparer des documents dont le vocabulaire diffère peu "
        "mais qui restent en anglais.",
    ),
    (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "Moyen (~470 Mo) — multilingue (bon pour le français), bon compromis vitesse/précision. "
        "À réserver au texte libre réellement rédigé en français avec des formulations variées "
        "(courriers, rapports) : sur des documents très gabarités, n'apporte pas d'avantage "
        "mesurable par rapport à TF-IDF ou à un modèle anglais, pour un coût de calcul bien plus élevé.",
    ),
    (
        "all-mpnet-base-v2",
        "Moyen-lourd (~420 Mo) — la meilleure qualité en anglais, plus lent. Pertinent sur un "
        "corpus anglophone hétérogène en texte libre ; sans intérêt particulier sur des documents "
        "administratifs très gabarités, quelle que soit leur langue.",
    ),
    (
        "paraphrase-multilingual-mpnet-base-v2",
        "Lourd (~970 Mo) — la meilleure qualité multilingue (français inclus), le plus lent. À "
        "réserver à un corpus français réellement hétérogène en texte libre (contrats de nature "
        "variée, correspondance...) où deux documents d'une même catégorie peuvent être formulés "
        "très différemment ; un coût inutile sur des documents très gabarités.",
    ),
]

STOPWORDS = sorted(set(sk_text.ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS)

# Ignore les jetons numériques et trop courts : bruit fréquent dans les PDF
# (dates, identifiants, chaînes encodées) qui polluait le nommage des catégories.
TOKEN_PATTERN = r"(?u)\b[^\W\d_][^\W\d_][^\W\d_]+\b"


class TfidfEngine:
    name = ENGINE_TFIDF

    def __init__(self, max_features: int | None = None, ngram_max: int | None = None):
        config = get_config()
        self.max_features = max_features if max_features is not None else config.tfidf_max_features
        ngram_max = ngram_max if ngram_max is not None else config.tfidf_ngram_max
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=TOKEN_PATTERN,
            stop_words=STOPWORDS,
            ngram_range=(1, ngram_max),
            min_df=1,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        return normalize(matrix).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return normalize(matrix).toarray()


class EmbeddingEngine:
    name = ENGINE_EMBEDDINGS

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Le moteur 'embeddings' nécessite le paquet optionnel "
                    "sentence-transformers : pip install -r requirements-embeddings.txt\n"
                    "Sinon, utilisez --engine tfidf pour rester 100% hors-ligne."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        embeddings = model.encode(texts, show_progress_bar=False)
        return normalize(embeddings)


def create_engine(
    name: str,
    embedding_model: str | None = None,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
):
    if name == ENGINE_TFIDF:
        return TfidfEngine(max_features=tfidf_max_features, ngram_max=tfidf_ngram_max)
    if name == ENGINE_EMBEDDINGS:
        return EmbeddingEngine(model_name=embedding_model or get_config().embedding_model_default)
    raise ValueError(f"Moteur inconnu : {name!r} (attendu : {ENGINE_TFIDF!r} ou {ENGINE_EMBEDDINGS!r})")


def engine_to_state(engine) -> dict:
    if isinstance(engine, TfidfEngine):
        return {"type": ENGINE_TFIDF, "vectorizer": engine.vectorizer}
    if isinstance(engine, EmbeddingEngine):
        return {"type": ENGINE_EMBEDDINGS, "model_name": engine.model_name}
    raise TypeError(f"Type de moteur non sérialisable : {type(engine)!r}")


def engine_from_state(state: dict):
    if state["type"] == ENGINE_TFIDF:
        engine = TfidfEngine()
        engine.vectorizer = state["vectorizer"]
        return engine
    if state["type"] == ENGINE_EMBEDDINGS:
        return EmbeddingEngine(model_name=state["model_name"])
    raise ValueError(f"État de moteur inconnu : {state['type']!r}")
