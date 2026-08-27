"""Point d'entrée CLI de l'outil de catégorisation de PDF.

Voir pdf_classifier/cli.py pour le détail des sous-commandes.

Exemples :
    python ml_pdf.py discover --input ./liste_pdf --output ./classified
    python ml_pdf.py train --input ./mes_categories --model model.pkl
    python ml_pdf.py classify --input ./nouveaux_pdf --model model.pkl --output ./classified
"""
from pdf_classifier.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
