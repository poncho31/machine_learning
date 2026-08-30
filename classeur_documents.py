"""Point d'entrée CLI de l'outil de catégorisation de documents.

Voir document_classifier/cli.py pour le détail des sous-commandes.

Exemples :
    python classeur_documents.py discover --input ./mes_documents --output ./classified
    python classeur_documents.py train --input ./mes_categories --model model.pkl
    python classeur_documents.py classify --input ./nouveaux_documents --model model.pkl --output ./classified
"""
from document_classifier.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
