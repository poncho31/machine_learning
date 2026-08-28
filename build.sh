#!/usr/bin/env bash
# Compile l'application en exécutable Linux autonome avec PyInstaller.
# Équivalent Linux de build.ps1 (voir ce fichier pour le détail des choix —
# abandon de Nuitka après plusieurs échecs liés à PyMuPDF/MinGW, PyInstaller
# copie les extensions C déjà compilées au lieu de les recompiler).
#
# PyInstaller ne fait PAS de compilation croisée : ce script doit être
# exécuté sur Linux (ou dans WSL) pour produire un binaire Linux, tout comme
# build.ps1 doit être exécuté sur Windows pour produire un .exe.
#
# Ce script écrit dans build-linux/, jamais dans build-windows/. NE JAMAIS
# lancer ce script en même temps que build.ps1 sur un dossier de sortie
# partagé (ex. un même dossier monté depuis Windows dans WSL).
#
# Usage :
#   ./build.sh              # variante lite, dossier (onedir)
#   ./build.sh --onefile    # variante lite, un seul exécutable
#   ./build.sh --full       # variante complète (embeddings)
#   ./build.sh --full --onefile

set -euo pipefail
cd "$(dirname "$0")"

VARIANT="lite"
ONEFILE=0
for arg in "$@"; do
    case "$arg" in
        --full) VARIANT="full" ;;
        --onefile) ONEFILE=1 ;;
        *) echo "Argument inconnu : $arg" >&2; exit 1 ;;
    esac
done

COMMON_ARGS=(
    "--noconfirm"
    "--clean"
    "--windowed"
    "--name=ClasseurDocuments"
    "--distpath=build-linux/dist"
    "--workpath=build-linux/work"
    "--specpath=build-linux"
)
# Contrairement à Windows, un exécutable Linux n'embarque pas d'icône : sur
# un bureau Linux, l'icône vient d'un fichier .desktop séparé qui référence
# une image (voir la doc si vous packagez pour un environnement de bureau).

if [ "$VARIANT" = "lite" ]; then
    for mod in torch torchvision torchaudio torchsde sentencepiece sentence_transformers transformers pandas matplotlib; do
        COMMON_ARGS+=("--exclude-module=$mod")
    done
fi

if [ "$ONEFILE" -eq 1 ]; then
    COMMON_ARGS+=("--onefile")
else
    COMMON_ARGS+=("--onedir")
fi

echo "Compilation ($VARIANT, $([ "$ONEFILE" -eq 1 ] && echo onefile || echo onedir)) avec PyInstaller..."
python3 -m PyInstaller "${COMMON_ARGS[@]}" ml_pdf_gui.py
