"""Configuration technique de l'application, pilotable via un fichier de
configuration (`config.json` à la racine du projet par défaut, modifiable
depuis l'onglet Paramètres de la GUI).

Toutes les valeurs ci-dessous ont un défaut raisonnable : le fichier de
configuration n'est nécessaire que pour les modifier. Les réglages bas
niveau qui pourraient casser l'extraction ou la vectorisation si mal réglés
(expression régulière de tokenisation, liste de mots vides...) restent dans
le code plutôt que d'être exposés ici.

`get_config()` met la configuration en cache après la première lecture ;
`reload_config()` la relit depuis le disque (appelé après un enregistrement
depuis l'onglet Paramètres) pour que les changements s'appliquent
immédiatement, sans redémarrer l'application.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields

from .utils import write_json_atomic

DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class AppConfig:
    # ── Regroupement automatique (onglet Entraînement) ──
    cluster_k_min: int = 2
    # Aucun plafond technique caché dans le code : c'est ici la seule limite
    # au nombre de catégories détectées automatiquement. Augmentez librement
    # cette valeur si vous avez besoin de plus de catégories.
    cluster_k_max: int = 20
    cluster_naming_top_words: int = 3
    # Score de silhouette minimal pour accepter un découpage en plusieurs
    # catégories (entre -1 et 1, la valeur qu'utilise scikit-learn). En
    # dessous de ce seuil pour TOUTES les valeurs de k testées entre
    # cluster_k_min et cluster_k_max, le regroupement automatique renonce à
    # découper et garde une seule catégorie.
    #
    # ATTENTION — ce score n'est PAS comparable d'un moteur à l'autre : mesuré
    # sur un vrai jeu de test (3 catégories de documents réellement distinctes),
    # TF-IDF n'atteint qu'un score d'environ 0.03 pour le DÉCOUPAGE CORRECT, là
    # où les embeddings atteignent 0.19 pour ce même découpage correct — TF-IDF
    # opère dans un espace très creux et de grande dimension où le score de
    # silhouette est structurellement bien plus bas, même quand la séparation
    # est réelle. Il n'existe PAS de seuil unique qui sépare fiablement le vrai
    # du bruit pour les deux moteurs (un découpage bruité peut même scorer plus
    # haut qu'un découpage correct avec TF-IDF sur un petit corpus). La valeur
    # par défaut ci-dessous est donc un compromis délibérément permissif —
    # pensé pour ne pas bloquer une vraie séparation avec TF-IDF — et non une
    # garantie : `cluster_min_cluster_size` ci-dessous reste la protection la
    # plus fiable contre un découpage dégénéré, et l'onglet Entraînement permet
    # d'ajuster ce seuil au cas par cas (voir ses préréglages) plutôt que de
    # dépendre d'une seule valeur globale pensée pour convenir à tous les cas.
    cluster_min_silhouette: float = 0.12
    # Nombre minimal de documents pour qu'une catégorie soit retenue lors de
    # la recherche automatique de k. Sans ce plancher, le score de silhouette
    # grimpe artificiellement à mesure que k se rapproche du nombre total de
    # documents : un cluster réduit à un seul document a toujours une
    # cohésion parfaite (aucune distance interne), ce qui gonfle son score
    # sans refléter une vraie catégorie — mesuré sur un petit corpus où le
    # score "optimal" grimpait jusqu'à k = presque tous les documents,
    # produisant une catégorie par document ou presque plutôt qu'une
    # catégorisation utilisable.
    cluster_min_cluster_size: int = 2

    # ── Vectorisation ──
    tfidf_max_features: int = 4000
    tfidf_ngram_max: int = 2
    embedding_model_default: str = "all-MiniLM-L6-v2"

    # ── Classification ──
    confidence_threshold: float = 0.4
    uncertain_category_name: str = "a_verifier"
    unreadable_category_name: str = "non_categorise_texte_illisible"
    # Catégorie fourre-tout où fusionner une catégorie supprimée dans
    # l'onglet Transformer les données (les documents ne sont jamais perdus).
    other_category_name: str = "autre"
    # État par défaut de la case "Améliorer le modèle avec ces documents"
    # dans l'onglet Classification (décochée par défaut).
    classification_improve_model_default: bool = False
    # État par défaut de la case "Inclure les fichiers 'à vérifier' dans
    # l'export" de l'onglet Classification (cochée par défaut : rien n'est
    # exclu de l'export sans action explicite de l'utilisateur).
    classification_export_uncertain_default: bool = True
    # Idem pour les fichiers "non catégorisé" (texte illisible).
    classification_export_unreadable_default: bool = True

    # ── Dossiers de sortie ──
    # Racine où chaque modèle créé depuis l'onglet Entraînement obtient son
    # propre dossier : storage/models/<nom>/<nom>.pkl, <nom>.json (fichiers
    # référencés) et dataset/<catégorie>/ (documents utilisés pour le
    # construire et l'améliorer). Toujours accessible même après un
    # redémarrage de l'application, sans dépendre d'un dossier temporaire.
    models_root: str = "storage/models"
    default_output_dir: str = "./classified"

    # ── Automatisation ──
    automation_config_path: str = "automations.json"
    automation_default_interval_value: int = 10
    automation_default_interval_unit: str = "minutes"
    automation_default_move: bool = True
    # État par défaut de la case "Inclure les fichiers 'à vérifier'" pour une
    # nouvelle automatisation (cochée par défaut : rien n'est exclu sans
    # action explicite de l'utilisateur).
    automation_default_include_uncertain: bool = True
    # Idem pour les fichiers "non catégorisé" (texte illisible).
    automation_default_include_unreadable: bool = True

    # ── Modèles locaux ──
    model_discovery_max_depth: int = 2
    # Nombre d'instantanés conservés par modèle (pkl_history/, json_history/,
    # dataset_history/) avant qu'un entraînement/amélioration/renommage ne
    # l'écrase — permet de revenir en arrière. 0 = aucun historique conservé.
    model_history_keep: int = 10

    # ── Interface ──
    window_width: int = 1050
    window_height: int = 700

    # ── API locale (onglet API) ──
    # N'écoute jamais que sur 127.0.0.1 (jamais exposée au réseau). Une clé
    # aléatoire, régénérée à chaque démarrage du serveur, est requise sur
    # toutes les routes sauf "/" et "/health".
    api_port: int = 8756


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not os.path.exists(path):
        return AppConfig()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known_fields = {field.name for field in fields(AppConfig)}
    filtered = {key: value for key, value in data.items() if key in known_fields}
    return AppConfig(**filtered)


def save_config(config: AppConfig, path: str = DEFAULT_CONFIG_PATH) -> None:
    write_json_atomic(asdict(config), path)


_current_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Configuration courante, chargée une seule fois puis mise en cache."""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config


def reload_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Relit le fichier de configuration — à appeler après un enregistrement
    pour que le reste de l'application voie immédiatement les changements."""
    global _current_config
    _current_config = load_config(path)
    return _current_config
