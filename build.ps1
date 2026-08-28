# Compile l'application en executable Windows autonome avec PyInstaller.
#
# Remplace un premier essai avec Nuitka, abandonné : PyMuPDF embarque un
# fichier C auto-généré de ~2,3 millions de lignes (module.mupdf.c) que MSVC
# ne peut pas compiler ("C1002 : espace du tas insuffisant"), et MinGW64 casse
# sur un problème d'en-têtes Windows différent, touchant même les fichiers
# coeur de Nuitka. PyInstaller copie les extensions C déjà compilées (le .pyd
# de PyMuPDF fourni par le paquet) au lieu de les recompiler depuis la
# source : il évite structurellement ce type d'erreur. L'exécutable produit
# est un peu plus gros et démarre un peu plus lentement qu'un Nuitka réussi
# (interprétation bytecode au lieu de code compilé), mais le build est
# nettement plus robuste avec cette pile de dépendances (numpy + scipy +
# scikit-learn + PyMuPDF).
#
# Deux variantes :
#   - "lite" (par defaut) : moteur TF-IDF uniquement, exclut explicitement
#     sentence-transformers/torch/pandas/matplotlib (jamais utilisés par
#     l'application) même s'ils sont installés dans l'environnement de build.
#   - "full" : inclut aussi le moteur embeddings (sentence-transformers +
#     torch) -- executable nettement plus gros (~1-2 Go).
#
# Usage :
#   powershell -File build.ps1              # variante lite, dossier (onedir)
#   powershell -File build.ps1 -Onefile      # variante lite, un seul .exe
#   powershell -File build.ps1 -Variant full # variante complete (embeddings)

param(
    [ValidateSet("lite", "full")]
    [string]$Variant = "lite",
    [switch]$Onefile
)

$ErrorActionPreference = "Stop"

$commonArgs = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name=ClasseurDocuments",
    "--icon=$(Resolve-Path assets/icon.ico)",
    "--distpath=build-windows/dist",
    "--workpath=build-windows/work",
    "--specpath=build-windows"
)

if ($Variant -eq "lite") {
    foreach ($mod in "torch", "torchvision", "torchaudio", "torchsde", "sentencepiece", "sentence_transformers", "transformers", "pandas", "matplotlib") {
        $commonArgs += "--exclude-module=$mod"
    }
}

if ($Onefile) {
    $commonArgs += "--onefile"
} else {
    $commonArgs += "--onedir"
}

Write-Host "Compilation ($Variant, $(if ($Onefile) {'onefile'} else {'onedir'})) avec PyInstaller..."
python -m PyInstaller @commonArgs ml_pdf_gui.py
