"""Moteurs de vectorisation du texte.

- TfidfEngine (par défaut) : aucune dépendance lourde, aucun téléchargement,
  fonctionne hors-ligne dès l'installation.
- EmbeddingEngine (optionnel) : meilleure qualité sémantique via
  sentence-transformers, nécessite un téléchargement de modèle (~90 Mo) au
  tout premier lancement puis fonctionne hors-ligne (modèle mis en cache).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import get_config

# scikit-learn avertit (à raison, en général) que `token_pattern` est ignoré
# dès qu'un `tokenizer` personnalisé est fourni — mais ici c'est TOUJOURS
# volontaire (voir TfidfEngine, mode racinisation) : `token_pattern` n'est
# jamais passé explicitement en même temps qu'un tokenizer, seul le défaut
# interne de scikit-learn (pas None) déclenche cet avertissement.
warnings.filterwarnings(
    "ignore",
    message=r"The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
    category=UserWarning,
)
from .stopwords_fr import FRENCH_STOPWORDS

ENGINE_TFIDF = "tfidf"
ENGINE_EMBEDDINGS = "embeddings"
ENGINE_IMAGE = "image"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"

# Catalogue de modèles CLIP (analyse VISUELLE — personnes, scènes, objets,
# pas le texte imprimé sur l'image, voir formats.ocr pour ça) via
# sentence-transformers (déjà une dépendance du moteur "embeddings", aucun
# paquet supplémentaire). Se télécharge une seule fois puis fonctionne
# hors-ligne, comme les modèles d'embeddings texte ci-dessous.
IMAGE_MODEL_CATALOG = [
    (
        "clip-ViT-B-32",
        "Léger (~350 Mo) — le plus rapide. Recommandé par défaut pour démarrer : suffisant pour "
        "distinguer des catégories visuelles nettement différentes (personnes, paysages, "
        "documents, intérieur/extérieur...).",
    ),
    (
        "clip-ViT-L-14",
        "Lourd (~890 Mo) — plus précis, plus lent. À réserver à un grand nombre de photos "
        "visuellement proches à départager finement (ex. beaucoup de photos de personnes à "
        "trier plus finement que juste \"avec/sans personne\").",
    ),
]

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

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("french")
except ImportError:
    _STEMMER = None


class _StemmingTokenizer:
    """Tokenizer TF-IDF avec racinisation (config `tfidf_use_stemming`),
    pour que des variantes d'un même mot ("facture"/"factures"/"facturé")
    deviennent un seul et même jeton au lieu de plusieurs jetons distincts
    qui diluent leur poids respectif. Racinisation française (Snowball) —
    ce projet cible en priorité des documents administratifs en français ;
    les mots déjà anglais ou très courts en ressortent peu ou pas modifiés.

    Filtre les mots vides STANDARDS avant de raciniser (mots grammaticaux
    très fréquents, non affectés par la racinisation) mais compare
    `extra_stopwords` APRÈS racinisation, aux racines déjà pré-racinées une
    fois pour toutes dans `__init__` : sinon exclure "fournisseur" laisserait
    passer "fournisseurs" une fois racinisé vers cette même forme.

    Une CLASSE avec un attribut simple plutôt qu'une closure sur
    `extra_stopwords` : un `.pkl` de modèle contient ce tokenizer tel quel
    (voir `engine_to_state`) — une closure locale n'est pas picklable, une
    instance de classe module-level avec un attribut l'est."""

    def __init__(self, extra_stopwords: frozenset[str] = frozenset()):
        self.extra_stopwords = extra_stopwords
        if extra_stopwords and _STEMMER is not None:
            self._extra_stemmed = frozenset(_STEMMER.stemWords(list(extra_stopwords)))
        else:
            self._extra_stemmed = extra_stopwords

    def __call__(self, text: str) -> list[str]:
        tokens = [t for t in re.findall(TOKEN_PATTERN, text) if t not in STOPWORDS]
        if _STEMMER is not None:
            tokens = _STEMMER.stemWords(tokens)
        if self._extra_stemmed:
            tokens = [t for t in tokens if t not in self._extra_stemmed]
        return tokens


class TfidfEngine:
    name = ENGINE_TFIDF

    def __init__(
        self,
        max_features: int | None = None,
        ngram_max: int | None = None,
        use_stemming: bool | None = None,
        extra_stopwords: frozenset[str] = frozenset(),
    ):
        config = get_config()
        self.max_features = max_features if max_features is not None else config.tfidf_max_features
        ngram_max = ngram_max if ngram_max is not None else config.tfidf_ngram_max
        use_stemming = use_stemming if use_stemming is not None else config.tfidf_use_stemming

        if use_stemming:
            # `stop_words`/`token_pattern` natifs de scikit-learn opèrent sur
            # des mots ENTIERS et sont ignorés dès qu'un `tokenizer`
            # personnalisé est fourni — `_StemmingTokenizer` fait donc lui-
            # même tout le travail (jetons + mots vides, extra_stopwords compris).
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                tokenizer=_StemmingTokenizer(extra_stopwords),
                ngram_range=(1, ngram_max),
                min_df=1,
            )
        else:
            stop_words = STOPWORDS if not extra_stopwords else sorted(set(STOPWORDS) | extra_stopwords)
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                token_pattern=TOKEN_PATTERN,
                stop_words=stop_words,
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

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, use_gpu: bool | None = None):
        self.model_name = model_name
        self.use_gpu = use_gpu if use_gpu is not None else get_config().embedding_use_gpu
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
            # GPU seulement s'il en détecte un compatible ET que le réglage
            # l'autorise — repli silencieux sur le CPU sinon (jamais d'erreur
            # juste parce qu'aucun GPU n'est présent sur la machine).
            device = None
            if self.use_gpu:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                except ImportError:
                    pass
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        embeddings = model.encode(texts, show_progress_bar=False)
        return normalize(embeddings)


class ImageEmbeddingEngine:
    """Vectorise des IMAGES par leur contenu visuel (personnes, scène,
    objets...) via un modèle CLIP — pas le texte qu'elles contiennent (voir
    `formats.py`, OCR, pour ça : une capacité totalement différente et
    indépendante). `fit_transform`/`transform` prennent des CHEMINS de
    fichiers image, pas du texte, contrairement à `TfidfEngine`/
    `EmbeddingEngine` — voir `discover._build_bundle`, qui bascule sur
    `doc.path` plutôt que `doc.text` pour ce moteur.

    CLIP projette aussi du texte dans ce même espace vectoriel : `encode_texts`
    permet de comparer un centroïde de cluster à une liste de libellés
    candidats (voir `discover._name_image_clusters`, nommage "zero-shot" —
    sans texte propre au document dont extraire des mots-clés, contrairement
    à TF-IDF/KeyBERT)."""

    name = ENGINE_IMAGE

    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL, use_gpu: bool | None = None):
        self.model_name = model_name
        self.use_gpu = use_gpu if use_gpu is not None else get_config().embedding_use_gpu
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Le moteur 'image' nécessite le paquet optionnel sentence-transformers : "
                    "pip install -r requirements-embeddings.txt"
                ) from exc
            device = None
            if self.use_gpu:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                except ImportError:
                    pass
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def _load_images(self, paths: list[str]):
        from PIL import Image

        return [Image.open(path).convert("RGB") for path in paths]

    def fit_transform(self, paths: list[str]) -> np.ndarray:
        return self.transform(paths)

    def transform(self, paths: list[str]) -> np.ndarray:
        model = self._load()
        images = self._load_images(paths)
        try:
            embeddings = model.encode(images, show_progress_bar=False)
        finally:
            for image in images:
                image.close()
        return normalize(embeddings)

    def encode_texts(self, labels: list[str]) -> np.ndarray:
        """Projette des libellés texte dans le MÊME espace vectoriel que les
        images (spécificité CLIP) — utilisé pour le nommage zero-shot des
        catégories, pas pour la vectorisation des documents eux-mêmes."""
        model = self._load()
        return normalize(model.encode(labels, show_progress_bar=False))


def create_engine(
    name: str,
    embedding_model: str | None = None,
    tfidf_max_features: int | None = None,
    tfidf_ngram_max: int | None = None,
    tfidf_use_stemming: bool | None = None,
    tfidf_extra_stopwords: frozenset[str] = frozenset(),
):
    if name == ENGINE_TFIDF:
        return TfidfEngine(
            max_features=tfidf_max_features, ngram_max=tfidf_ngram_max,
            use_stemming=tfidf_use_stemming, extra_stopwords=tfidf_extra_stopwords,
        )
    if name == ENGINE_EMBEDDINGS:
        return EmbeddingEngine(model_name=embedding_model or get_config().embedding_model_default)
    if name == ENGINE_IMAGE:
        return ImageEmbeddingEngine(model_name=embedding_model or get_config().image_model_default)
    raise ValueError(
        f"Moteur inconnu : {name!r} (attendu : {ENGINE_TFIDF!r}, {ENGINE_EMBEDDINGS!r} ou {ENGINE_IMAGE!r})"
    )


def engine_to_state(engine) -> dict:
    if isinstance(engine, TfidfEngine):
        return {"type": ENGINE_TFIDF, "vectorizer": engine.vectorizer}
    if isinstance(engine, ImageEmbeddingEngine):
        return {"type": ENGINE_IMAGE, "model_name": engine.model_name}
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
    if state["type"] == ENGINE_IMAGE:
        return ImageEmbeddingEngine(model_name=state["model_name"])
    raise ValueError(f"État de moteur inconnu : {state['type']!r}")
