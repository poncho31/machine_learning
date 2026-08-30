"""Interface en ligne de commande.

Exemples :
    python classeur_documents.py discover --input ./mes_documents --output ./classified

    python classeur_documents.py train --input ./mes_categories --model model.pkl

    python classeur_documents.py classify --input ./nouveaux_documents --model model.pkl --output ./classified
"""
from __future__ import annotations

import argparse
import sys

from .classify import classify as classify_fn
from .discover import discover as discover_fn
from .features import ENGINE_EMBEDDINGS, ENGINE_TFIDF
from .train import train as train_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classeur_documents", description="Catégorisation locale de documents, sans LLM et sans connexion internet."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser(
        "discover", help="Regroupe des documents non triés par similarité (non supervisé, sans exemples au préalable)."
    )
    p_discover.add_argument("--input", required=True, help="Dossier contenant les documents à regrouper.")
    p_discover.add_argument("--output", required=True, help="Dossier de destination des documents classés.")
    p_discover.add_argument("--engine", choices=[ENGINE_TFIDF, ENGINE_EMBEDDINGS], default=ENGINE_TFIDF)
    p_discover.add_argument("--embedding-model", default=None, help="Nom du modèle sentence-transformers (si --engine embeddings).")
    p_discover.add_argument("--min-clusters", type=int, default=2)
    p_discover.add_argument("--max-clusters", type=int, default=10)
    p_discover.add_argument("--recursive", action="store_true", help="Cherche aussi dans les sous-dossiers.")
    p_discover.add_argument("--move", action="store_true", help="Déplace les fichiers au lieu de les copier.")
    p_discover.add_argument("--model-out", default=None, help="Sauvegarde ce regroupement comme modèle réutilisable avec 'classify'.")
    p_discover.add_argument("--tfidf-max-features", type=int, default=None, help="Nombre max de termes TF-IDF (défaut : config).")
    p_discover.add_argument("--tfidf-ngram-max", type=int, default=None, help="Taille max des n-grammes TF-IDF (défaut : config).")
    p_discover.add_argument(
        "--stemming", dest="tfidf_use_stemming", action="store_true", default=None,
        help="Racinise les mots (FR/EN) avant la vectorisation TF-IDF (défaut : config).",
    )
    p_discover.add_argument(
        "--no-stemming", dest="tfidf_use_stemming", action="store_false",
        help="Désactive la racinisation même si activée par défaut dans la config.",
    )
    p_discover.add_argument(
        "--use-keybert", action="store_true", default=None,
        help="Nomme les catégories via KeyBERT plutôt que par mots-clés TF-IDF (défaut : config).",
    )
    p_discover.add_argument(
        "--detect-duplicates", action="store_true", default=None,
        help="Écarte les documents quasi identiques avant le regroupement (défaut : config).",
    )
    p_discover.add_argument(
        "--cluster-algorithm", choices=["kmeans", "minibatch_kmeans", "agglomerative", "hdbscan"], default=None,
        help="Algorithme de regroupement (défaut : config `cluster_algorithm`).",
    )
    p_discover.add_argument(
        "--cluster-svd", dest="cluster_use_svd", action="store_true", default=None,
        help="Réduit la dimension des vecteurs TF-IDF (SVD/LSA) avant regroupement (défaut : config).",
    )
    p_discover.add_argument(
        "--no-cluster-svd", dest="cluster_use_svd", action="store_false",
        help="Désactive la réduction SVD même si activée par défaut dans la config.",
    )
    p_discover.add_argument("--cluster-svd-components", type=int, default=None, help="Nombre de composantes SVD (défaut : config).")
    p_discover.add_argument(
        "--cluster-extra-stopwords", default=None,
        help="Mots supplémentaires à ignorer, séparés par des virgules (défaut : config).",
    )

    p_train = sub.add_parser(
        "train", help="Entraîne un modèle à partir de documents déjà triés par l'utilisateur dans des sous-dossiers (= catégories)."
    )
    p_train.add_argument("--input", required=True, help="Dossier contenant un sous-dossier par catégorie, rempli manuellement.")
    p_train.add_argument("--model", required=True, help="Chemin du modèle à créer (.pkl).")
    p_train.add_argument("--engine", choices=[ENGINE_TFIDF, ENGINE_EMBEDDINGS], default=ENGINE_TFIDF)
    p_train.add_argument("--embedding-model", default=None, help="Nom du modèle sentence-transformers (si --engine embeddings).")

    p_classify = sub.add_parser("classify", help="Classe de nouveaux documents avec un modèle déjà entraîné.")
    p_classify.add_argument("--input", required=True, help="Dossier contenant les nouveaux documents à classer.")
    p_classify.add_argument("--model", required=True, help="Modèle entraîné (.pkl) à utiliser.")
    p_classify.add_argument("--output", required=True, help="Dossier de destination des documents classés.")
    p_classify.add_argument("--threshold", type=float, default=0.4, help="Confiance minimale ; en dessous, le fichier va dans 'a_verifier'.")
    p_classify.add_argument("--recursive", action="store_true", help="Cherche aussi dans les sous-dossiers.")
    p_classify.add_argument("--move", action="store_true", help="Déplace les fichiers au lieu de les copier.")

    return parser


def main(argv: list[str] | None = None) -> int:
    # La console Windows utilise par défaut un encodage (cp1252) qui ne
    # supporte pas les accents/flèches affichés pendant le traitement ; on
    # force l'UTF-8 pour que l'outil marche tel quel, sans configuration
    # côté utilisateur.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "discover":
            discover_fn(
                input_dir=args.input,
                output_dir=args.output,
                engine_name=args.engine,
                embedding_model=args.embedding_model,
                k_min=args.min_clusters,
                k_max=args.max_clusters,
                recursive=args.recursive,
                move=args.move,
                model_out=args.model_out,
                tfidf_max_features=args.tfidf_max_features,
                tfidf_ngram_max=args.tfidf_ngram_max,
                tfidf_use_stemming=args.tfidf_use_stemming,
                use_keybert=args.use_keybert,
                detect_duplicates=args.detect_duplicates,
                cluster_algorithm=args.cluster_algorithm,
                cluster_use_svd=args.cluster_use_svd,
                cluster_svd_components=args.cluster_svd_components,
                cluster_extra_stopwords=args.cluster_extra_stopwords,
            )
        elif args.command == "train":
            train_fn(
                labeled_dir=args.input,
                model_path=args.model,
                engine_name=args.engine,
                embedding_model=args.embedding_model,
            )
        elif args.command == "classify":
            classify_fn(
                input_dir=args.input,
                model_path=args.model,
                output_dir=args.output,
                threshold=args.threshold,
                recursive=args.recursive,
                move=args.move,
            )
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
