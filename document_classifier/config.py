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
    # Utilise KeyBERT (mots-clés choisis par similarité sémantique via un
    # modèle d'embeddings, plutôt que par simple fréquence TF-IDF) pour
    # nommer les catégories détectées — noms généralement plus naturels et
    # lisibles. Réutilise all-MiniLM-L6-v2, déjà le modèle d'embeddings léger
    # par défaut de l'application. Se replie automatiquement sur le nommage
    # TF-IDF si le paquet optionnel `keybert` n'est pas installé.
    cluster_use_keybert: bool = True
    # Détecte, pendant l'entraînement, les paires de documents quasi
    # identiques (similarité cosinus sur les vecteurs déjà calculés pour le
    # regroupement — aucun calcul supplémentaire lourd). Désactivé
    # automatiquement au-delà de `cluster_duplicate_max_docs` documents pour
    # éviter un coût en O(n²) sur un très gros corpus.
    cluster_detect_duplicates: bool = True
    cluster_duplicate_threshold: float = 0.97
    cluster_duplicate_max_docs: int = 4000
    # État par défaut de la case "Réécrire la configuration" de l'onglet
    # Entraînement (section "Modèle existant à améliorer") : cochée par
    # défaut, elle recharge dans le formulaire les réglages EXACTS
    # (training_params) que le modèle sélectionné avait lors de son dernier
    # entraînement, plutôt que de laisser le formulaire sur des valeurs
    # potentiellement différentes (un préréglage choisi entre-temps, par
    # exemple) qui écraseraient silencieusement la configuration d'origine.
    train_rewrite_config_default: bool = True
    # Algorithme de regroupement. "kmeans" (défaut) recherche automatiquement
    # le meilleur k entre cluster_k_min/max ; "minibatch_kmeans" est une
    # variante approximative bien plus rapide sur un gros corpus (voir
    # cluster_large_corpus_threshold, qui bascule dessus automatiquement même
    # si "kmeans" est sélectionné) ; "agglomerative" (hiérarchique, linkage
    # de Ward) donne souvent des groupes plus cohérents sur un corpus petit à
    # moyen mais est plus lent ; "hdbscan" ne force PAS chaque document dans
    # un cluster — les documents trop atypiques restent "non classés"
    # (fusionnés dans la catégorie "autre") plutôt que rattachés
    # arbitrairement au cluster le moins pire, mais ne respecte pas
    # cluster_k_min/max (le nombre de groupes est déduit des données).
    cluster_algorithm: str = "kmeans"
    # Au-delà de ce nombre de documents lisibles, "kmeans" bascule
    # automatiquement sur "minibatch_kmeans" (résultat très proche, beaucoup
    # plus rapide) — sans ça, un entraînement sur un très gros dossier peut
    # devenir extrêmement lent (KMeans plein recalcule sur tout le corpus à
    # chaque itération). Les autres algorithmes ne sont pas concernés.
    cluster_large_corpus_threshold: int = 2000
    # Réduit la dimension des vecteurs TF-IDF (SVD tronquée, "LSA") avant le
    # regroupement — l'espace TF-IDF est très creux et de grande dimension,
    # ce qui structurellement écrase le score de silhouette même pour une
    # séparation réelle (voir la discussion dans discover.py) ; une
    # projection sur moins de dimensions resserre les distances et peut
    # améliorer la qualité du découpage. Sans effet pour le moteur
    # embeddings, déjà dense et de dimension raisonnable.
    cluster_use_svd: bool = True
    cluster_svd_components: int = 200
    # Calcule, en plus du score de silhouette déjà utilisé pour choisir le
    # nombre de catégories, deux métriques complémentaires (Davies-Bouldin :
    # plus bas est meilleur ; Calinski-Harabasz : plus haut est meilleur),
    # affichées dans le journal d'entraînement — utile pour croiser plusieurs
    # indices plutôt que dépendre d'un seul.
    cluster_report_extra_metrics: bool = True
    # Mots à ignorer en plus des mots vides FR/EN habituels lors du nommage
    # des catégories et de la vectorisation TF-IDF (ex. le nom de
    # l'entreprise, qui apparaît dans chaque document et pollue sinon le
    # nommage) — séparés par des virgules. Valeur de départ du formulaire de
    # l'onglet Entraînement, modifiable pour CET entraînement précis sans
    # changer ce défaut (comme les autres réglages avancés).
    cluster_extra_stopwords: str = ""

    # ── Vectorisation ──
    tfidf_max_features: int = 4000
    tfidf_ngram_max: int = 2
    # Racinise les mots (stemming FR/EN, ex. "facture"/"factures"/"facturé"
    # -> "factur") avant de les compter pour le TF-IDF, pour que des variantes
    # d'un même mot ne soient plus des jetons distincts qui diluent leur
    # poids respectif. Se replie automatiquement sur les mots tels quels si
    # le paquet optionnel `snowballstemmer` n'est pas installé.
    tfidf_use_stemming: bool = True
    embedding_model_default: str = "all-MiniLM-L6-v2"
    # Utilise le GPU (CUDA) pour le moteur embeddings s'il en détecte un
    # disponible — nettement plus rapide sur un gros corpus. Sans effet (et
    # sans erreur) si aucun GPU compatible n'est détecté : repli silencieux
    # sur le CPU.
    embedding_use_gpu: bool = True
    # Modèle CLIP utilisé par le moteur "image" (analyse VISUELLE — personnes,
    # scènes, objets — pas le texte imprimé sur l'image, voir `ocr_enabled`
    # pour ça). Se télécharge une fois puis fonctionne hors-ligne, comme
    # `embedding_model_default`. Réutilise `embedding_use_gpu` ci-dessus.
    image_model_default: str = "clip-ViT-B-32"
    # Libellés candidats comparés à chaque catégorie d'images détectée (nom
    # "zero-shot" via CLIP — il n'y a pas de texte propre au document dont
    # extraire des mots-clés comme pour TF-IDF/KeyBERT) : le libellé le plus
    # proche du contenu visuel moyen du groupe devient son nom. Séparés par
    # des virgules, modifiable pour un entraînement précis (comme
    # `cluster_extra_stopwords`) sans changer ce défaut.
    image_cluster_labels: str = (
        "personnes, groupe de personnes, portrait, paysage, intérieur, extérieur, "
        "document texte, écran, objet, véhicule, animal, nourriture, événement ou fête, "
        "architecture ou bâtiment"
    )

    # ── Classification ──
    confidence_threshold: float = 0.4
    uncertain_category_name: str = "a_verifier"
    unreadable_category_name: str = "non_categorise_texte_illisible"
    # Catégorie fourre-tout où fusionner une catégorie supprimée dans
    # l'onglet Classification (les documents ne sont jamais perdus).
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

    # ── Extraction ──
    # Nombre de documents extraits EN PARALLÈLE (un thread par fichier, via
    # concurrent.futures — l'extraction est dominée par l'attente I/O disque,
    # pas par le calcul, donc de vrais threads suffisent sans les limites du
    # GIL). 1 = séquentiel (comportement d'avant, utile pour déboguer un
    # fichier problématique en isolant les messages d'erreur).
    extraction_parallel_workers: int = 4

    # ── OCR (PDF scannés et images) ──
    # Active la reconnaissance de caractères (Tesseract) pour les PDF sans
    # texte extractible et les images (.png, .jpg, .jpeg, .tiff, .bmp) —
    # désactivé par défaut : plus lent qu'une extraction directe. Le paquet
    # Python pytesseract est installé par défaut (requirements.txt), mais le
    # moteur Tesseract lui-même reste à installer séparément sur la machine
    # (pas un paquet pip — voir le README, section "Formats de fichiers pris
    # en charge"). Sans Tesseract installé, ces fichiers restent simplement
    # non pris en charge.
    ocr_enabled: bool = False
    # Chemin de l'exécutable Tesseract si absent du PATH (courant sur
    # Windows) — laissez vide s'il est déjà trouvé automatiquement.
    tesseract_cmd_path: str = ""

    # ── Interface ──
    window_width: int = 1050
    window_height: int = 700
    # Dernier modèle chargé dans l'onglet Classification — présélectionné
    # automatiquement au prochain lancement de l'application, pour ne pas
    # avoir à le rechoisir à chaque fois. Vide = aucun modèle à présélectionner.
    last_model_path: str = ""

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
