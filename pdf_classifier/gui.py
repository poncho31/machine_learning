"""Interface graphique complète : classification, entraînement et
automatisation, dans une même fenêtre à onglets.

Le vrai glisser-déposer depuis l'Explorateur Windows nécessite le paquet
optionnel `tkinterdnd2`. S'il n'est pas installé, l'onglet Classification
reste pleinement utilisable via les boutons "Ajouter des fichiers..." /
"Ajouter un dossier...".
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from . import api_server, model_store
from .automation import AutomationConfig, AutomationManager
from .classify import (
    known_categories,
    load_model_for_prediction,
    predict_labels,
    uncertain_category,
    unreadable_category,
)
from .config import DEFAULT_CONFIG_PATH, AppConfig, get_config, reload_config, save_config
from .discover import build_model as build_model_fn
from .discover import improve_model as improve_model_fn
from .extraction import ExtractedDocument, extract_text, list_documents
from .features import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_CATALOG, ENGINE_EMBEDDINGS, ENGINE_TFIDF
from .formats import SUPPORTED_EXTENSIONS, is_supported
from .rename import delete_category, list_category_files, rename_categories, rename_files_with_prefix
from .utils import dispatch_file, write_json_atomic

FILETYPES_DOCS = [
    ("Documents pris en charge", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
    ("Tous les fichiers", "*.*"),
]


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} To"


# ── Onglet Classification ────────────────────────────────────────
class Row:
    """Un document en cours de revue dans le tableau."""

    def __init__(self, path: str):
        self.path = path
        self.filename = os.path.basename(path)
        self.text = ""
        self.predicted_category = ""
        self.confidence: float | None = None
        self.corrected_category = ""
        # True seulement si l'utilisateur a explicitement changé la catégorie
        # via le double-clic — pas quand elle vient simplement de la
        # prédiction (même à forte confiance) ou du repli automatique
        # "a_verifier". Sert à ne faire contribuer à l'amélioration continue
        # du modèle que les corrections humaines réelles.
        self.manually_corrected = False
        self.item_id: str | None = None


class ClassifyTab(ttk.Frame):
    def __init__(self, parent, on_model_improved=None):
        super().__init__(parent, padding=10)

        self.on_model_improved = on_model_improved
        self.bundle = None
        self.engine = None
        self.model_path = tk.StringVar(value="(aucun modèle chargé)")
        self.output_dir = tk.StringVar(value=os.path.abspath(get_config().default_output_dir))
        self.move_files = tk.BooleanVar(value=False)
        self.improve_model_var = tk.BooleanVar(value=get_config().classification_improve_model_default)
        self.export_uncertain_var = tk.BooleanVar(value=get_config().classification_export_uncertain_default)
        self.export_unreadable_var = tk.BooleanVar(value=get_config().classification_export_unreadable_default)
        self.rows: dict[str, Row] = {}
        self.discovered_models: list[tuple[str, int]] = []
        self.selected_row: Row | None = None
        self.last_dispatch_dir: str | None = None

        self._build_layout()
        self._refresh_model_picker()

    def _build_layout(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")

        picker_row = ttk.Frame(top)
        picker_row.pack(fill="x")
        ttk.Label(picker_row, text="Modèles disponibles (du plus léger au plus lourd) :").pack(side="left")
        self.model_picker_var = tk.StringVar()
        self.model_picker = ttk.Combobox(picker_row, textvariable=self.model_picker_var, state="readonly", width=55)
        self.model_picker.pack(side="left", padx=6)
        self.model_picker.bind("<<ComboboxSelected>>", self._on_pick_model)
        ttk.Button(picker_row, text="Rafraîchir", command=self._refresh_model_picker).pack(side="left")

        browse_row = ttk.Frame(top)
        browse_row.pack(fill="x", pady=(4, 0))
        ttk.Button(browse_row, text="Charger un modèle (.pkl)...", command=self._choose_model).pack(side="left")
        ttk.Label(browse_row, textvariable=self.model_path, foreground="#555").pack(side="left", padx=8)

        drop_frame = ttk.Frame(self)
        drop_frame.pack(fill="x")

        drop_text = (
            "Glissez des documents ici, ou utilisez les boutons ci-dessous"
            if _DND_AVAILABLE
            else "Glisser-déposer indisponible (pip install tkinterdnd2) — utilisez les boutons ci-dessous"
        )
        self.drop_label = tk.Label(drop_frame, text=drop_text, bg="#eef2f7", fg="#33475b", relief="ridge", height=3)
        self.drop_label.pack(fill="x", pady=6)

        if _DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Ajouter des fichiers...", command=self._add_files_dialog).pack(side="left")
        ttk.Button(buttons, text="Ajouter un dossier...", command=self._add_folder_dialog).pack(side="left", padx=6)
        ttk.Button(buttons, text="Vider la liste", command=self._clear_rows).pack(side="left")

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=10)

        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        columns = ("filename", "predicted", "confidence", "corrected")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("filename", text="Fichier")
        self.tree.heading("predicted", text="Catégorie proposée")
        self.tree.heading("confidence", text="Confiance")
        self.tree.heading("corrected", text="Catégorie retenue")
        self.tree.column("filename", width=320)
        self.tree.column("predicted", width=180)
        self.tree.column("confidence", width=90, anchor="center")
        self.tree.column("corrected", width=180)
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", self._edit_selected_category)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="left", fill="y")

        self.tree.tag_configure("uncertain", background="#fff3cd")
        self.tree.tag_configure("unreadable", background="#f8d7da")

        preview_frame = ttk.Frame(paned, padding=(10, 0, 0, 0))
        paned.add(preview_frame, weight=2)

        ttk.Label(preview_frame, text="Aperçu du texte analysé :").pack(anchor="w")
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap="word", state="disabled", height=10)
        self.preview_text.pack(fill="both", expand=True, pady=(2, 6))
        self.open_file_button = ttk.Button(
            preview_frame, text="Ouvrir le fichier original", command=self._open_selected_file, state="disabled"
        )
        self.open_file_button.pack(anchor="w")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x")

        ttk.Label(bottom, text="Dossier de sortie :").grid(row=0, column=0, sticky="w")
        ttk.Entry(bottom, textvariable=self.output_dir, width=60).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(bottom, text="Parcourir...", command=self._choose_output_dir).grid(row=0, column=2)
        bottom.columnconfigure(1, weight=1)

        ttk.Checkbutton(bottom, text="Déplacer les fichiers (au lieu de copier)", variable=self.move_files).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.validate_button = ttk.Button(bottom, text="Valider et classer", command=self._validate)
        self.validate_button.grid(row=1, column=2, pady=(6, 0))

        ttk.Checkbutton(
            bottom,
            text="Inclure les fichiers \"à vérifier\" dans l'export",
            variable=self.export_uncertain_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.open_dispatch_button = ttk.Button(
            bottom, text="Ouvrir le dernier dossier classé", command=self._open_last_dispatch_dir, state="disabled"
        )
        self.open_dispatch_button.grid(row=2, column=2, pady=(2, 0))

        ttk.Checkbutton(
            bottom,
            text="Inclure les fichiers \"non catégorisé\" dans l'export",
            variable=self.export_unreadable_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        ttk.Checkbutton(
            bottom,
            text="Améliorer le modèle avec ces documents (catégories corrigées prises comme référence)",
            variable=self.improve_model_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.improve_progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.improve_progress.grid(row=5, column=0, columnspan=3, sticky="we", pady=(6, 0))

        self.status = tk.StringVar(value="Chargez un modèle pour commencer.")
        ttk.Label(self, textvariable=self.status, foreground="#555").pack(fill="x", pady=(4, 0))

    # ── Modèle ──
    def _refresh_model_picker(self) -> None:
        self.discovered_models = model_store.discover_models(".")
        display_values = [
            f"{os.path.relpath(path)}  ({_human_size(size)})" for path, size in self.discovered_models
        ]
        self.model_picker.configure(values=display_values)
        if not display_values:
            self.model_picker_var.set("(aucun modèle .pkl trouvé dans le projet)")

    def _on_pick_model(self, _event=None) -> None:
        index = self.model_picker.current()
        if index < 0 or index >= len(self.discovered_models):
            return
        path, _size = self.discovered_models[index]
        self._load_model(path)

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(title="Choisir un modèle", filetypes=[("Modèle entraîné", "*.pkl")])
        if not path:
            return
        self._load_model(path)

    def _load_model(self, path: str) -> None:
        try:
            self.bundle, self.engine = load_model_for_prediction(path)
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle :\n{exc}")
            return
        self.model_path.set(path)
        categories = known_categories(self.bundle)
        self.status.set(f"Modèle chargé ({self.bundle['mode']}) — {len(categories)} catégorie(s) connue(s).")

    def reload_if_active(self, model_path: str) -> None:
        """Recharge le modèle actuellement chargé s'il vient d'être renommé
        dans l'onglet Transformer les données, pour refléter les nouveaux noms."""
        if self.bundle is not None and os.path.abspath(self.model_path.get()) == os.path.abspath(model_path):
            self._load_model(model_path)

    # ── Ajout de fichiers ──
    def _add_files_dialog(self) -> None:
        paths = filedialog.askopenfilenames(title="Choisir des documents", filetypes=FILETYPES_DOCS)
        self._add_paths(paths)

    def _add_folder_dialog(self) -> None:
        directory = filedialog.askdirectory(title="Choisir un dossier de documents")
        if not directory:
            return
        self._add_paths(list_documents(directory))

    def _on_drop(self, event) -> None:
        paths = self.tk.splitlist(event.data)
        documents = []
        for path in paths:
            if os.path.isdir(path):
                documents.extend(list_documents(path, recursive=True))
            elif is_supported(path):
                documents.append(path)
        self._add_paths(documents)

    def _add_paths(self, paths) -> None:
        new_paths = [p for p in paths if p and p not in self.rows]
        if not new_paths:
            return
        if self.bundle is None:
            messagebox.showwarning("Aucun modèle", "Chargez un modèle avant d'ajouter des fichiers.")
            return
        for path in new_paths:
            self.rows[path] = Row(path)
        self.status.set(f"Analyse de {len(new_paths)} fichier(s)...")
        threading.Thread(target=self._predict_paths, args=(new_paths,), daemon=True).start()

    def _predict_paths(self, paths: list[str]) -> None:
        threshold = get_config().confidence_threshold
        for path in paths:
            row = self.rows[path]
            text, _error = extract_text(path)
            row.text = text
            if not text.strip():
                row.predicted_category = unreadable_category()
                row.confidence = 0.0
                # Le fichier n'a pas pu être lu : ce n'est pas une prédiction
                # incertaine, donc pas un cas "a_verifier" — le seuil de
                # confiance ne s'applique pas ici.
                row.corrected_category = unreadable_category()
            else:
                vectors = self.engine.transform([text])
                labels, confidences = predict_labels(self.bundle, vectors)
                row.predicted_category = labels[0]
                row.confidence = float(confidences[0])
                row.corrected_category = (
                    row.predicted_category if row.confidence >= threshold else uncertain_category()
                )
            self.after(0, self._insert_row, row)
        self.after(0, lambda: self.status.set(f"{len(self.rows)} fichier(s) au total."))

    def _insert_row(self, row: Row) -> None:
        confidence_display = f"{row.confidence:.0%}" if row.confidence is not None else "n/a"
        tag = ()
        if row.predicted_category == unreadable_category():
            tag = ("unreadable",)
        elif row.confidence is not None and row.confidence < get_config().confidence_threshold:
            tag = ("uncertain",)
        row.item_id = self.tree.insert(
            "", "end",
            values=(row.filename, row.predicted_category, confidence_display, row.corrected_category),
            tags=tag,
        )

    def _clear_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.selected_row = None
        self._set_preview_text("")
        self.open_file_button.configure(state="disabled")
        self.status.set("Liste vidée.")

    # ── Aperçu du fichier sélectionné ──
    def _on_select_row(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            self.selected_row = None
            self._set_preview_text("")
            self.open_file_button.configure(state="disabled")
            return
        row = next((r for r in self.rows.values() if r.item_id == selection[0]), None)
        self.selected_row = row
        if row is None:
            return
        self._set_preview_text(row.text if row.text.strip() else "(aucun texte n'a pu être extrait de ce fichier)")
        self.open_file_button.configure(state="normal")

    def _set_preview_text(self, text: str) -> None:
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def _open_selected_file(self) -> None:
        if self.selected_row and os.path.isfile(self.selected_row.path):
            os.startfile(self.selected_row.path)

    # ── Correction manuelle ──
    def _edit_selected_category(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        item_id = selection[0]
        row = next((r for r in self.rows.values() if r.item_id == item_id), None)
        if row is None:
            return

        categories = known_categories(self.bundle) if self.bundle else []
        categories = sorted(set(categories) | {row.predicted_category, uncertain_category()})

        popup = tk.Toplevel(self)
        popup.title(f"Catégorie — {row.filename}")
        popup.geometry("320x100")
        ttk.Label(popup, text="Catégorie retenue :").pack(padx=10, pady=(10, 2), anchor="w")
        var = tk.StringVar(value=row.corrected_category)
        combo = ttk.Combobox(popup, textvariable=var, values=categories)
        combo.pack(fill="x", padx=10)

        def confirm() -> None:
            new_category = var.get().strip()
            if new_category and new_category != row.corrected_category:
                row.corrected_category = new_category
                row.manually_corrected = True
                self.tree.set(row.item_id, "corrected", new_category)
            popup.destroy()

        ttk.Button(popup, text="OK", command=confirm).pack(pady=8)
        combo.focus_set()

    # ── Validation ──
    def _choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Dossier de sortie")
        if directory:
            self.output_dir.set(directory)

    def _validate(self) -> None:
        if not self.rows:
            messagebox.showinfo("Rien à faire", "Aucun fichier à classer.")
            return
        output_dir = self.output_dir.get().strip()
        if not output_dir:
            messagebox.showwarning("Dossier manquant", "Choisissez un dossier de sortie.")
            return

        # Chaque validation crée son propre sous-dossier horodaté (comme
        # l'aperçu d'entraînement) : rien n'est jamais écrasé d'un clic à l'autre.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(output_dir, f"classification_{timestamp}")

        include_uncertain = self.export_uncertain_var.get()
        include_unreadable = self.export_unreadable_var.get()
        uncertain_name = uncertain_category()
        unreadable_name = unreadable_category()

        results = {}
        errors = []
        skipped_uncertain = 0
        skipped_unreadable = 0
        # Seules les lignes réellement corrigées à la main (double-clic) et
        # dont la nouvelle catégorie n'est pas un simple repli ("a_verifier",
        # "illisible") contribuent à l'amélioration continue du modèle — pas
        # une catégorie confirmée simplement parce qu'elle n'a pas été
        # touchée, même à forte confiance.
        improve_batch: list[ExtractedDocument] = []
        confirmed_labels: dict[str, str] = {}
        placeholder_categories = {uncertain_category(), unreadable_category()}
        model_path_snapshot = self.model_path.get()

        for row in self.rows.values():
            category = row.corrected_category or uncertain_category()
            if category == uncertain_name and not include_uncertain:
                skipped_uncertain += 1
                continue
            if category == unreadable_name and not include_unreadable:
                skipped_unreadable += 1
                continue
            try:
                dest = dispatch_file(row.path, category, run_dir, move=self.move_files.get())
                results[row.path] = {
                    "category": category,
                    "predicted_category": row.predicted_category,
                    "confidence": row.confidence,
                    "destination": dest,
                }
                if row.manually_corrected and category not in placeholder_categories and row.text.strip():
                    improve_batch.append(ExtractedDocument(path=dest, filename=os.path.basename(dest), text=row.text))
                    confirmed_labels[dest] = category
            except Exception as exc:
                errors.append(f"{row.filename} : {exc}")

        if results:
            write_json_atomic(results, os.path.join(run_dir, "classification.json"))
            self.last_dispatch_dir = run_dir
            self.open_dispatch_button.configure(state="normal")

        should_improve = self.improve_model_var.get() and self.bundle is not None and improve_batch
        self._clear_rows()

        message = f"{len(results)} fichier(s) classé(s) dans {run_dir}."
        if skipped_uncertain:
            message += f"\n{skipped_uncertain} fichier(s) \"à vérifier\" exclu(s) de l'export."
        if skipped_unreadable:
            message += f"\n{skipped_unreadable} fichier(s) \"non catégorisé\" exclu(s) de l'export."
        if errors:
            message += "\n\nErreurs :\n" + "\n".join(errors)
            messagebox.showwarning("Terminé avec erreurs", message)
        else:
            messagebox.showinfo("Terminé", message)
        self.status.set(f"{len(results)} fichier(s) classé(s).")

        if results and os.path.isdir(run_dir):
            os.startfile(run_dir)

        if should_improve:
            self._start_model_improvement(model_path_snapshot, improve_batch, confirmed_labels)

    def _open_last_dispatch_dir(self) -> None:
        if self.last_dispatch_dir and os.path.isdir(self.last_dispatch_dir):
            os.startfile(self.last_dispatch_dir)

    # ── Amélioration continue du modèle ──
    def _start_model_improvement(
        self, model_path: str, documents: list[ExtractedDocument], confirmed_labels: dict[str, str]
    ) -> None:
        self.validate_button.configure(state="disabled")
        self.improve_progress.start(12)
        self.status.set("Amélioration du modèle en cours...")
        threading.Thread(
            target=self._run_model_improvement, args=(model_path, documents, confirmed_labels), daemon=True
        ).start()

    def _run_model_improvement(
        self, model_path: str, documents: list[ExtractedDocument], confirmed_labels: dict[str, str]
    ) -> None:
        def progress(message: str) -> None:
            self.after(0, self.status.set, message)

        try:
            improve_model_fn(model_path, documents, confirmed_labels, progress=progress)
            self.after(0, self._on_model_improved_done, model_path)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Erreur", f"Amélioration du modèle impossible :\n{exc}"))
        finally:
            self.after(0, self.improve_progress.stop)
            self.after(0, lambda: self.validate_button.configure(state="normal"))

    def _on_model_improved_done(self, model_path: str) -> None:
        self._load_model(model_path)
        self.status.set("Modèle amélioré avec les documents validés.")
        if self.on_model_improved:
            self.on_model_improved(model_path)


# ── Onglet Entraînement ───────────────────────────────────────────
ENGINE_EXPLANATION = (
    "TF-IDF (par défaut) : compare les documents par les mots qu'ils ont en commun. "
    "Rapide, disponible immédiatement (aucun téléchargement), très efficace quand chaque "
    "catégorie a un vocabulaire distinct (factures, contrats, relevés fiscaux...).\n\n"
    "Embeddings (sémantique) : compare les documents par le SENS des phrases plutôt que "
    "les mots exacts, via un modèle de langage pré-entraîné choisi ci-dessous, du plus léger "
    "au plus lourd. Plus précis quand deux documents parlent de la même chose avec des mots "
    "différents, mais plus lent et nécessite un téléchargement au tout premier usage.\n\n"
    "Mesuré sur un vrai jeu de documents très gabarités (même mise en page, vocabulaire quasi "
    "identique d'un exemplaire à l'autre — factures d'un même fournisseur, fiches de paie...) : "
    "les embeddings n'y apportent aucun gain, y compris les modèles multilingues sur des "
    "documents en français — la variation d'un exemplaire à l'autre y est numérique (dates, "
    "montants), pas sémantique. Dans ce cas, restez sur TF-IDF. Les embeddings creusent l'écart "
    "sur un corpus réellement hétérogène en texte libre, où deux documents d'une même catégorie "
    "peuvent être formulés très différemment.\n\n"
    "Le score qui décide s'il existe assez de séparation pour découper (silhouette) n'a PAS la "
    "même échelle selon le moteur : sur un même découpage réellement correct, TF-IDF ne scorait "
    "qu'environ 0.03 dans nos mesures contre 0.19 pour les embeddings — un seuil unique ne peut "
    "donc pas être parfait pour les deux. Utilisez les préréglages et paramètres avancés ci-dessous "
    "pour ajuster ce compromis à votre cas plutôt que de dépendre d'une seule valeur par défaut."
)

# Chaque préréglage fixe une combinaison cohérente de moteur/seuils pour un
# cas d'usage donné, plutôt que de laisser deviner quels réglages toucher —
# voir la discussion sur l'incomparabilité du score de silhouette entre
# moteurs (ENGINE_EXPLANATION ci-dessus) : un dictionnaire vide reprend les
# valeurs actuelles de Paramètres, sans rien imposer.
TRAINING_PRESETS: list[tuple[str, str, dict]] = [
    (
        "Par défaut (valeurs de Paramètres)",
        "Aucun réglage particulier : reprend les valeurs actuelles de l'onglet Paramètres.",
        {},
    ),
    (
        "Documents très gabarités (même modèle, ex. factures d'un fournisseur, fiches de paie)",
        "TF-IDF, peu de catégories attendues, un nombre minimal de documents par catégorie plus "
        "élevé pour éviter de découper sur de simples variations d'un exemplaire à l'autre.",
        {
            "engine": ENGINE_TFIDF, "k_max": 6,
            "cluster_min_silhouette": 0.15, "cluster_min_cluster_size": 3,
        },
    ),
    (
        "Documents hétérogènes en texte libre (contrats variés, courriers, rapports)",
        "Embeddings multilingues : plus adaptés à des documents qui parlent de la même chose "
        "sans partager le même vocabulaire exact.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "k_max": 15, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Mélange de plusieurs types de documents (factures + contrats + fiches de paie...)",
        "TF-IDF avec un vocabulaire plus large : des types de documents différents partagent en "
        "général peu de vocabulaire exact, TF-IDF les sépare bien sans rien télécharger.",
        {
            "engine": ENGINE_TFIDF, "tfidf_max_features": 6000, "tfidf_ngram_max": 2,
            "k_max": 20, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Exploration permissive (tout faire ressortir, à trier ensuite)",
        "Seuils très bas : fait ressortir un maximum de sous-groupes, y compris incertains, à "
        "corriger ensuite dans l'onglet Transformer les données.",
        {"k_max": 20, "cluster_min_silhouette": 0.02, "cluster_min_cluster_size": 1},
    ),
]
_EMBEDDING_DESCRIPTIONS = dict(EMBEDDING_MODEL_CATALOG)


class TrainTab(ttk.Frame):
    """Aucun tri manuel requis : on pointe un dossier de documents en vrac,
    les catégories sont détectées automatiquement (regroupement par
    similarité), comme le mode 'discover' de la CLI."""

    def __init__(self, parent, on_model_created=None):
        super().__init__(parent, padding=10)
        self.on_model_created = on_model_created
        self.input_dir = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.base_model_path = tk.StringVar()
        self.engine = tk.StringVar(value=ENGINE_TFIDF)
        self.embedding_model = tk.StringVar(value=DEFAULT_EMBEDDING_MODEL)
        self.preset_var = tk.StringVar(value=TRAINING_PRESETS[0][0])
        config = get_config()
        self.k_min_var = tk.IntVar(value=config.cluster_k_min)
        self.k_max_var = tk.IntVar(value=config.cluster_k_max)
        self.min_silhouette_var = tk.DoubleVar(value=config.cluster_min_silhouette)
        self.min_cluster_size_var = tk.IntVar(value=config.cluster_min_cluster_size)
        self.tfidf_max_features_var = tk.IntVar(value=config.tfidf_max_features)
        self.tfidf_ngram_max_var = tk.IntVar(value=config.tfidf_ngram_max)
        self.last_model_path: str | None = None
        self._build()
        self.model_name_var.trace_add("write", self._update_resolved_path)

    def _build(self) -> None:
        ttk.Label(self, text="1. Dossier de documents à analyser (en vrac, aucun tri préalable requis) :").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Entry(self, textvariable=self.input_dir, width=70).grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Button(self, text="Parcourir...", command=self._choose_input_dir).grid(row=1, column=2, padx=4)

        ttk.Label(self, text="2. Moteur d'analyse :").grid(row=2, column=0, sticky="w", pady=(14, 0))
        engine_combo = ttk.Combobox(
            self, textvariable=self.engine, values=[ENGINE_TFIDF, ENGINE_EMBEDDINGS], state="readonly", width=15
        )
        engine_combo.grid(row=3, column=0, sticky="w")
        engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

        embedding_frame = ttk.Frame(self)
        embedding_frame.grid(row=3, column=1, columnspan=2, sticky="w", padx=(20, 0))
        self.embedding_label = ttk.Label(embedding_frame, text="Modèle d'embeddings (léger → lourd) :")
        self.embedding_label.pack(anchor="w")
        self.embedding_combo = ttk.Combobox(
            embedding_frame, textvariable=self.embedding_model,
            values=[name for name, _description in EMBEDDING_MODEL_CATALOG],
            state="readonly", width=45,
        )
        self.embedding_combo.pack(anchor="w")
        self.embedding_combo.bind("<<ComboboxSelected>>", self._update_embedding_description)

        self.embedding_description_var = tk.StringVar()
        ttk.Label(
            embedding_frame, textvariable=self.embedding_description_var,
            foreground="#555", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        self._update_embedding_description()

        explanation = tk.Label(
            self, text=ENGINE_EXPLANATION, justify="left", wraplength=760, anchor="w",
            bg="#eef2f7", fg="#33475b", padx=8, pady=6,
        )
        explanation.grid(row=4, column=0, columnspan=3, sticky="we", pady=(6, 0))

        self._build_advanced_params()
        self._on_engine_change()

        ttk.Label(
            self, text="3. Modèle existant à améliorer avec ces nouvelles données (optionnel) :"
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Entry(self, textvariable=self.base_model_path, width=70).grid(row=7, column=0, columnspan=2, sticky="we")
        ttk.Button(self, text="Parcourir...", command=self._choose_base_model).grid(row=7, column=2, padx=4)
        ttk.Button(self, text="Aucun (nouveau modèle)", command=lambda: self.base_model_path.set("")).grid(
            row=8, column=0, sticky="w", pady=(2, 0)
        )

        ttk.Label(self, text="4. Nom du modèle :").grid(row=9, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Entry(self, textvariable=self.model_name_var, width=40).grid(row=10, column=0, sticky="w")
        self.resolved_path_var = tk.StringVar()
        ttk.Label(self, textvariable=self.resolved_path_var, foreground="#555").grid(
            row=10, column=1, columnspan=2, sticky="w", padx=(8, 0)
        )

        action_row = ttk.Frame(self)
        action_row.grid(row=11, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.train_button = ttk.Button(action_row, text="Entraîner le modèle", command=self._start_training)
        self.train_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            action_row, text="Ouvrir le dossier du modèle", command=self._open_preview_dir, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=6)

        self.progress_bar = ttk.Progressbar(self, mode="indeterminate")
        self.progress_bar.grid(row=12, column=0, columnspan=3, sticky="we", pady=(0, 8))

        self.log = scrolledtext.ScrolledText(self, height=14, state="disabled")
        self.log.grid(row=13, column=0, columnspan=3, sticky="nsew")

        self.grid_rowconfigure(13, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _on_engine_change(self, _event=None) -> None:
        is_embeddings = self.engine.get() == ENGINE_EMBEDDINGS
        self.embedding_combo.configure(state="readonly" if is_embeddings else "disabled")
        tfidf_state = "disabled" if is_embeddings else "normal"
        self.tfidf_max_features_spin.configure(state=tfidf_state)
        self.tfidf_ngram_max_spin.configure(state=tfidf_state)

    def _update_embedding_description(self, _event=None) -> None:
        self.embedding_description_var.set(_EMBEDDING_DESCRIPTIONS.get(self.embedding_model.get(), ""))

    def _build_advanced_params(self) -> None:
        frame = ttk.LabelFrame(self, text="Paramètres avancés du moteur (facultatif)", padding=10)
        frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=(10, 0))

        ttk.Label(frame, text="Préréglage :").grid(row=0, column=0, sticky="w")
        preset_combo = ttk.Combobox(
            frame, textvariable=self.preset_var,
            values=[name for name, _description, _values in TRAINING_PRESETS],
            state="readonly", width=62,
        )
        preset_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=(6, 0))
        preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)

        self.preset_description_var = tk.StringVar(value=TRAINING_PRESETS[0][1])
        ttk.Label(
            frame, textvariable=self.preset_description_var, foreground="#555",
            wraplength=720, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 8))

        def spin(row, col, label, var, from_, to, increment=1):
            ttk.Label(frame, text=label).grid(row=row, column=col * 2, sticky="w", padx=(0 if col == 0 else 16, 0))
            box = ttk.Spinbox(frame, from_=from_, to=to, increment=increment, textvariable=var, width=8)
            box.grid(row=row, column=col * 2 + 1, sticky="w", padx=(4, 0))
            return box

        spin(2, 0, "Nb minimal de catégories :", self.k_min_var, 1, 100)
        spin(2, 1, "Nb maximal de catégories :", self.k_max_var, 1, 200)
        spin(3, 0, "Score de silhouette minimal :", self.min_silhouette_var, 0.0, 1.0, 0.01)
        spin(3, 1, "Documents minimum par catégorie :", self.min_cluster_size_var, 1, 50)
        self.tfidf_max_features_spin = spin(4, 0, "Vocabulaire TF-IDF maximal :", self.tfidf_max_features_var, 100, 50000, 100)
        self.tfidf_ngram_max_spin = spin(4, 1, "Taille max des n-grammes :", self.tfidf_ngram_max_var, 1, 4)

        ttk.Label(
            frame,
            text="Ces valeurs s'appliquent uniquement à cet entraînement — les valeurs par défaut de "
            "l'application (onglet Paramètres) ne sont pas modifiées.",
            foreground="#555", wraplength=720, justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def _apply_preset(self, _event=None) -> None:
        name = self.preset_var.get()
        match = next((p for p in TRAINING_PRESETS if p[0] == name), None)
        if match is None:
            return
        description, values = match[1], match[2]
        self.preset_description_var.set(description)

        config = get_config()
        self.engine.set(values.get("engine", ENGINE_TFIDF))
        self.embedding_model.set(values.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        self.k_min_var.set(values.get("k_min", config.cluster_k_min))
        self.k_max_var.set(values.get("k_max", config.cluster_k_max))
        self.min_silhouette_var.set(values.get("cluster_min_silhouette", config.cluster_min_silhouette))
        self.min_cluster_size_var.set(values.get("cluster_min_cluster_size", config.cluster_min_cluster_size))
        self.tfidf_max_features_var.set(values.get("tfidf_max_features", config.tfidf_max_features))
        self.tfidf_ngram_max_var.set(values.get("tfidf_ngram_max", config.tfidf_ngram_max))

        self._on_engine_change()
        self._update_embedding_description()

    def _choose_input_dir(self) -> None:
        directory = filedialog.askdirectory(title="Dossier de documents")
        if directory:
            self.input_dir.set(directory)

    def _choose_base_model(self) -> None:
        path = filedialog.askopenfilename(title="Modèle existant à améliorer", filetypes=[("Modèle entraîné", "*.pkl")])
        if not path:
            return
        self.base_model_path.set(path)
        if not self.model_name_var.get().strip():
            # Par défaut, on améliore ce modèle en place : même nom.
            self.model_name_var.set(os.path.splitext(os.path.basename(path))[0])

    def _update_resolved_path(self, *_args) -> None:
        name = self.model_name_var.get().strip()
        if name:
            path = model_store.model_path_for_name(name)
            self.resolved_path_var.set(f"→ {path}")
        else:
            self.resolved_path_var.set("")

    def _log_line(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start_training(self) -> None:
        input_dir = self.input_dir.get().strip()
        model_name = self.model_name_var.get().strip()
        if not input_dir or not model_name:
            messagebox.showwarning("Champs manquants", "Choisissez le dossier de documents et le nom du modèle.")
            return
        model_path = model_store.model_path_for_name(model_name)

        self.train_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress_bar.start(12)

        engine_name = self.engine.get()
        embedding_model = self.embedding_model.get().strip() if engine_name == ENGINE_EMBEDDINGS else None
        base_model_path = self.base_model_path.get().strip() or None
        overrides = {
            "k_min": self.k_min_var.get(),
            "k_max": self.k_max_var.get(),
            "min_silhouette": self.min_silhouette_var.get(),
            "min_cluster_size": self.min_cluster_size_var.get(),
            "tfidf_max_features": self.tfidf_max_features_var.get(),
            "tfidf_ngram_max": self.tfidf_ngram_max_var.get(),
        }
        threading.Thread(
            target=self._run_training,
            args=(input_dir, model_path, engine_name, embedding_model, base_model_path, overrides),
            daemon=True,
        ).start()

    def _run_training(
        self, input_dir: str, model_path: str, engine_name: str, embedding_model: str | None,
        base_model_path: str | None, overrides: dict,
    ) -> None:
        def progress(message: str) -> None:
            self.after(0, self._log_line, message)

        try:
            build_model_fn(
                input_dir=input_dir,
                model_path=model_path,
                engine_name=engine_name,
                embedding_model=embedding_model,
                base_model_path=base_model_path,
                progress=progress,
                **overrides,
            )
            self.after(0, self._enable_open_folder, model_path)
            if self.on_model_created:
                self.after(0, self.on_model_created, model_path)
        except Exception as exc:
            self.after(0, self._log_line, f"\nErreur : {exc}")
        finally:
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.train_button.configure(state="normal"))

    def _enable_open_folder(self, model_path: str) -> None:
        self.last_model_path = model_path
        self.open_folder_button.configure(state="normal")

    def _open_preview_dir(self) -> None:
        if not self.last_model_path:
            return
        dataset_dir = model_store.model_dataset_dir(self.last_model_path)
        if os.path.isdir(dataset_dir):
            os.startfile(dataset_dir)


# ── Onglet Automatisation ─────────────────────────────────────────
class AutomationDialog(tk.Toplevel):
    """Boîte de dialogue d'ajout/modification d'une automatisation."""

    def __init__(self, parent, on_submit, existing: AutomationConfig | None = None):
        super().__init__(parent)
        self.title("Nouvelle automatisation" if existing is None else f"Modifier — {existing.name}")
        self.on_submit = on_submit
        self.existing = existing
        self.resizable(False, False)

        config = get_config()
        self.name_var = tk.StringVar(value=existing.name if existing else "")
        self.watch_dir = tk.StringVar(value=existing.watch_dir if existing else "")
        self.model_path = tk.StringVar(value=existing.model_path if existing else "")
        self.output_dir = tk.StringVar(value=existing.output_dir if existing else "")
        self.interval_value = tk.IntVar(
            value=existing.interval_value if existing else config.automation_default_interval_value
        )
        self.interval_unit = tk.StringVar(
            value=existing.interval_unit if existing else config.automation_default_interval_unit
        )
        self.move = tk.BooleanVar(value=existing.move if existing else config.automation_default_move)
        self.threshold = tk.DoubleVar(value=existing.threshold if existing else config.confidence_threshold)
        self.recursive = tk.BooleanVar(value=existing.recursive if existing else False)
        self.include_uncertain = tk.BooleanVar(
            value=existing.include_uncertain if existing else config.automation_default_include_uncertain
        )
        self.include_unreadable = tk.BooleanVar(
            value=existing.include_unreadable if existing else config.automation_default_include_unreadable
        )

        self._build()
        self.grab_set()

    def _build(self) -> None:
        pad = dict(padx=10, pady=4, sticky="w")
        row = 0
        ttk.Label(self, text="Nom :").grid(row=row, column=0, **pad)
        name_entry = ttk.Entry(self, textvariable=self.name_var, width=42)
        name_entry.grid(row=row, column=1, columnspan=2, **pad)
        if self.existing is not None:
            name_entry.configure(state="disabled")
        row += 1

        ttk.Label(self, text="Dossier surveillé :").grid(row=row, column=0, **pad)
        ttk.Entry(self, textvariable=self.watch_dir, width=42).grid(row=row, column=1, **pad)
        ttk.Button(self, text="Parcourir...", command=lambda: self._browse_dir(self.watch_dir)).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Modèle (.pkl) :").grid(row=row, column=0, **pad)
        ttk.Entry(self, textvariable=self.model_path, width=42).grid(row=row, column=1, **pad)
        ttk.Button(self, text="Parcourir...", command=self._browse_model).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Dossier de sortie :").grid(row=row, column=0, **pad)
        ttk.Entry(self, textvariable=self.output_dir, width=42).grid(row=row, column=1, **pad)
        ttk.Button(self, text="Parcourir...", command=lambda: self._browse_dir(self.output_dir)).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Intervalle :").grid(row=row, column=0, **pad)
        interval_frame = ttk.Frame(self)
        interval_frame.grid(row=row, column=1, columnspan=2, **pad)
        ttk.Spinbox(interval_frame, from_=1, to=10000, textvariable=self.interval_value, width=8).pack(side="left")
        ttk.Combobox(
            interval_frame, textvariable=self.interval_unit, values=["minutes", "heures", "jours"],
            state="readonly", width=10,
        ).pack(side="left", padx=6)
        row += 1

        ttk.Checkbutton(
            self,
            text="Déplacer les fichiers (recommandé : sinon ils sont reclassés à chaque passage tant qu'ils restent dans le dossier)",
            variable=self.move,
        ).grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        ttk.Checkbutton(self, text="Inclure les sous-dossiers", variable=self.recursive).grid(
            row=row, column=0, columnspan=3, **pad
        )
        row += 1

        ttk.Checkbutton(
            self,
            text="Inclure les fichiers \"à vérifier\" (sinon laissés en place, retentés au passage suivant)",
            variable=self.include_uncertain,
        ).grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        ttk.Checkbutton(
            self,
            text="Inclure les fichiers \"non catégorisé\" (sinon laissés en place, retentés au passage suivant)",
            variable=self.include_unreadable,
        ).grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        ttk.Label(self, text="Seuil de confiance minimal :").grid(row=row, column=0, **pad)
        ttk.Spinbox(self, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold, width=8).grid(
            row=row, column=1, **pad
        )
        row += 1

        buttons = ttk.Frame(self)
        buttons.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(buttons, text="Annuler", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Enregistrer", command=self._submit).pack(side="left")

    def _browse_dir(self, var: tk.StringVar) -> None:
        directory = filedialog.askdirectory(parent=self)
        if directory:
            var.set(directory)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(parent=self, filetypes=[("Modèle entraîné", "*.pkl")])
        if path:
            self.model_path.set(path)

    def _submit(self) -> None:
        name = self.name_var.get().strip()
        watch_dir = self.watch_dir.get().strip()
        model_path = self.model_path.get().strip()
        output_dir = self.output_dir.get().strip()
        if not all([name, watch_dir, model_path, output_dir]):
            messagebox.showwarning("Champs manquants", "Tous les champs sont requis.", parent=self)
            return

        config = AutomationConfig(
            name=name,
            watch_dir=watch_dir,
            model_path=model_path,
            output_dir=output_dir,
            interval_value=max(1, self.interval_value.get()),
            interval_unit=self.interval_unit.get(),
            move=self.move.get(),
            threshold=self.threshold.get(),
            recursive=self.recursive.get(),
            include_uncertain=self.include_uncertain.get(),
            include_unreadable=self.include_unreadable.get(),
            enabled=True,
        )
        try:
            self.on_submit(config, replacing=self.existing is not None)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc), parent=self)
            return
        self.destroy()


def _format_snapshot_timestamp(timestamp: str) -> str:
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S_%f").strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return timestamp


class HistoryDialog(tk.Toplevel):
    """Liste les instantanés disponibles pour un modèle (pris automatiquement
    avant chaque entraînement/amélioration/renommage) et permet d'y revenir.
    Restaurer archive d'abord l'état courant : la restauration elle-même
    peut donc être annulée en restaurant l'instantané créé juste avant."""

    def __init__(self, parent, model_path: str, on_restored=None):
        super().__init__(parent)
        self.model_path = model_path
        self.on_restored = on_restored
        self.timestamps: list[str] = []
        self.title(f"Historique — {os.path.basename(model_path)}")
        self.geometry("420x340")
        self.resizable(False, True)

        ttk.Label(
            self, text="Chaque entraînement, amélioration ou renommage archive l'état précédent.",
            wraplength=390, justify="left",
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self.listbox = tk.Listbox(self)
        self.listbox.pack(fill="both", expand=True, padx=10, pady=4)
        self._refresh()

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(buttons, text="Restaurer cet instantané", command=self._restore_selected).pack(side="left")
        ttk.Button(buttons, text="Fermer", command=self.destroy).pack(side="left", padx=6)

        self.grab_set()

    def _refresh(self) -> None:
        self.timestamps = model_store.list_snapshots(self.model_path)
        self.listbox.delete(0, "end")
        if not self.timestamps:
            self.listbox.insert("end", "(aucun instantané pour l'instant)")
            return
        for timestamp in self.timestamps:
            self.listbox.insert("end", _format_snapshot_timestamp(timestamp))

    def _restore_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection or not self.timestamps or selection[0] >= len(self.timestamps):
            return
        timestamp = self.timestamps[selection[0]]
        if not messagebox.askyesno(
            "Confirmer",
            "Revenir à cet instantané ?\n\n"
            "L'état actuel sera lui-même archivé avant : vous pourrez annuler "
            "cette restauration si besoin.",
            parent=self,
        ):
            return
        try:
            model_store.restore_snapshot(self.model_path, timestamp)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc), parent=self)
            return
        if self.on_restored:
            self.on_restored(self.model_path)
        self._refresh()
        messagebox.showinfo("Terminé", "Modèle restauré.", parent=self)


# ── Onglet Transformer les données ─────────────────────────────────
class TransformTab(ttk.Frame):
    """Permet de gérer les catégories détectées automatiquement par l'onglet
    Entraînement : les renommer, voir et renommer leurs fichiers, ou
    supprimer une catégorie (ses documents sont alors regroupés dans la
    catégorie "autre" — rien n'est jamais perdu). S'applique au modèle
    lui-même (les classifications futures verront ces changements) et à son
    dossier dataset/ (storage/models/<nom>/dataset/), toujours accessible
    même après un redémarrage de l'application."""

    def __init__(self, parent, on_renamed=None):
        super().__init__(parent, padding=10)
        self.on_renamed = on_renamed
        self.model_path = tk.StringVar(value="(aucun modèle chargé)")
        self.dataset_dir: str | None = None
        self.bundle: dict | None = None
        self.discovered_models: list[tuple[str, int]] = []
        self.selected_cluster_id: int | None = None
        self._build()
        self._refresh_model_picker()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")

        picker_row = ttk.Frame(top)
        picker_row.pack(fill="x")
        ttk.Label(picker_row, text="Modèle à modifier (du plus léger au plus lourd) :").pack(side="left")
        self.model_picker_var = tk.StringVar()
        self.model_picker = ttk.Combobox(picker_row, textvariable=self.model_picker_var, state="readonly", width=55)
        self.model_picker.pack(side="left", padx=6)
        self.model_picker.bind("<<ComboboxSelected>>", self._on_pick_model)
        ttk.Button(picker_row, text="Rafraîchir", command=self._refresh_model_picker).pack(side="left")

        browse_row = ttk.Frame(top)
        browse_row.pack(fill="x", pady=(4, 0))
        ttk.Button(browse_row, text="Charger un modèle (.pkl)...", command=self._choose_model).pack(side="left")
        self.history_button = ttk.Button(
            browse_row, text="Historique / Revenir en arrière...", command=self._open_history_dialog, state="disabled"
        )
        self.history_button.pack(side="left", padx=6)
        ttk.Label(browse_row, textvariable=self.model_path, foreground="#555").pack(side="left", padx=8)

        ttk.Label(self, text="Catégories détectées :").pack(anchor="w", pady=(16, 6))

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        tree_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=2)
        self.tree = ttk.Treeview(
            tree_frame, columns=("name", "detected", "count"), show="headings", selectmode="browse"
        )
        self.tree.heading("name", text="Catégorie")
        self.tree.heading("detected", text="Nom détecté par le modèle")
        self.tree.heading("count", text="Fichiers")
        self.tree.column("name", width=200)
        self.tree.column("detected", width=200)
        self.tree.column("count", width=70, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_category)

        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        tree_scrollbar.pack(side="left", fill="y")

        details = ttk.Frame(paned, padding=(10, 0, 0, 0))
        paned.add(details, weight=3)

        rename_row = ttk.Frame(details)
        rename_row.pack(fill="x")
        ttk.Label(rename_row, text="Nouveau nom :").pack(side="left")
        self.rename_var = tk.StringVar()
        self.rename_entry = ttk.Entry(rename_row, textvariable=self.rename_var, width=32, state="disabled")
        self.rename_entry.pack(side="left", padx=6)
        self.rename_button = ttk.Button(rename_row, text="Renommer", command=self._rename_selected, state="disabled")
        self.rename_button.pack(side="left")

        ttk.Label(details, text="Fichiers de cette catégorie :").pack(anchor="w", pady=(12, 2))
        files_frame = ttk.Frame(details)
        files_frame.pack(fill="both", expand=True)
        self.files_listbox = tk.Listbox(files_frame)
        self.files_listbox.pack(side="left", fill="both", expand=True)
        files_scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)
        files_scrollbar.pack(side="left", fill="y")

        actions = ttk.Frame(details)
        actions.pack(fill="x", pady=10)
        self.open_folder_button = ttk.Button(
            actions, text="Ouvrir le dossier", command=self._open_category_folder, state="disabled"
        )
        self.open_folder_button.pack(side="left")
        self.prefix_button = ttk.Button(
            actions, text="Préfixer les fichiers par la catégorie", command=self._prefix_files, state="disabled"
        )
        self.prefix_button.pack(side="left", padx=6)
        self.delete_button = ttk.Button(
            actions, text="Supprimer cette catégorie (→ autre)", command=self._delete_selected, state="disabled"
        )
        self.delete_button.pack(side="left")

        self.status = tk.StringVar(value="Chargez un modèle pour voir ses catégories.")
        ttk.Label(self, textvariable=self.status, foreground="#555").pack(fill="x", pady=(8, 0))

    # ── Modèle ──
    def _refresh_model_picker(self) -> None:
        self.discovered_models = model_store.discover_models(".")
        display_values = [
            f"{os.path.relpath(path)}  ({_human_size(size)})" for path, size in self.discovered_models
        ]
        self.model_picker.configure(values=display_values)

    def _on_pick_model(self, _event=None) -> None:
        index = self.model_picker.current()
        if index < 0 or index >= len(self.discovered_models):
            return
        path, _size = self.discovered_models[index]
        self._load_model(path)

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(title="Choisir un modèle", filetypes=[("Modèle entraîné", "*.pkl")])
        if path:
            self._load_model(path)

    def _open_history_dialog(self) -> None:
        model_path = self.model_path.get()
        if not model_path or not os.path.exists(model_path):
            return
        HistoryDialog(self, model_path, on_restored=self._on_history_restored)

    def _on_history_restored(self, model_path: str) -> None:
        self._load_model(model_path)
        if self.on_renamed:
            self.on_renamed(model_path)

    def _load_model(self, path: str) -> None:
        try:
            bundle = model_store.load_bundle(path)
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle :\n{exc}")
            return
        if bundle.get("mode") != "unsupervised":
            messagebox.showwarning(
                "Non pris en charge",
                "Seuls les modèles créés par l'onglet Entraînement (catégories détectées "
                "automatiquement) peuvent être modifiés ici.",
            )
            return
        self.model_path.set(path)
        self.bundle = bundle
        self.dataset_dir = model_store.model_dataset_dir(path)
        self.history_button.configure(state="normal")
        self._populate_categories()

    # ── Liste des catégories ──
    def _populate_categories(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if not self.bundle:
            self._clear_selection_ui()
            return

        cluster_names = self.bundle["cluster_names"]
        original_names = self.bundle.get("original_cluster_names", {})
        has_dataset = bool(self.dataset_dir and os.path.isdir(self.dataset_dir))
        for cluster_id, name in sorted(cluster_names.items()):
            count = len(list_category_files(self.dataset_dir, name)) if has_dataset else None
            detected_name = original_names.get(cluster_id, "n/a")
            self.tree.insert(
                "", "end", iid=str(cluster_id),
                values=(name, detected_name, count if count is not None else "n/a"),
            )

        dataset_note = (
            " (dossier dataset trouvé — fichiers consultables)"
            if has_dataset
            else " (dossier dataset pas encore créé — entraînez ou améliorez ce modèle une première fois)"
        )
        self.status.set(f"{len(cluster_names)} catégorie(s){dataset_note}.")
        self._clear_selection_ui()

    def _clear_selection_ui(self) -> None:
        self.selected_cluster_id = None
        self.rename_var.set("")
        self.rename_entry.configure(state="disabled")
        self.rename_button.configure(state="disabled")
        self.files_listbox.delete(0, "end")
        self.open_folder_button.configure(state="disabled")
        self.prefix_button.configure(state="disabled")
        self.delete_button.configure(state="disabled")

    def _selected_name(self) -> str | None:
        if self.selected_cluster_id is None or not self.bundle:
            return None
        return self.bundle["cluster_names"].get(self.selected_cluster_id)

    def _on_select_category(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection or not self.bundle:
            self._clear_selection_ui()
            return

        cluster_id = int(selection[0])
        self.selected_cluster_id = cluster_id
        name = self.bundle["cluster_names"][cluster_id]

        self.rename_var.set(name)
        self.rename_entry.configure(state="normal")
        self.rename_button.configure(state="normal")

        self.files_listbox.delete(0, "end")
        has_dataset = bool(self.dataset_dir and os.path.isdir(self.dataset_dir))
        if has_dataset:
            for filename in list_category_files(self.dataset_dir, name):
                self.files_listbox.insert("end", filename)
        self.open_folder_button.configure(state="normal" if has_dataset else "disabled")
        self.prefix_button.configure(state="normal" if has_dataset else "disabled")

        other_name = get_config().other_category_name
        can_delete = len(self.bundle["cluster_names"]) > 1 and name != other_name
        self.delete_button.configure(state="normal" if can_delete else "disabled")

    # ── Actions ──
    def _rename_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        new_name = self.rename_var.get().strip()
        if not new_name:
            messagebox.showwarning("Nom vide", "Le nom ne peut pas être vide.")
            return
        if new_name == name:
            return
        try:
            bundle = rename_categories(self.model_path.get(), {name: new_name})
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = bundle
        self._populate_categories()
        if self.on_renamed:
            self.on_renamed(self.model_path.get())
        self.status.set(f"Catégorie renommée : « {name} » → « {new_name} ».")

    def _open_category_folder(self) -> None:
        name = self._selected_name()
        if name is None or not self.dataset_dir:
            return
        folder = os.path.join(self.dataset_dir, name)
        if os.path.isdir(folder):
            os.startfile(folder)

    def _prefix_files(self) -> None:
        name = self._selected_name()
        if name is None or not self.dataset_dir:
            return
        count = rename_files_with_prefix(self.model_path.get(), name)
        self._on_select_category()  # rafraîchit la liste des fichiers affichés
        if count:
            self.status.set(f"{count} fichier(s) préfixé(s) par « {name} ».")
        else:
            self.status.set("Tous les fichiers étaient déjà préfixés.")

    def _delete_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        other_name = get_config().other_category_name
        if not messagebox.askyesno(
            "Confirmer",
            f"Supprimer la catégorie « {name} » ?\n\n"
            f"Ses documents seront regroupés dans « {other_name} » — rien n'est perdu.",
        ):
            return
        try:
            bundle = delete_category(self.model_path.get(), name, other_name=other_name)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = bundle
        self._populate_categories()
        if self.on_renamed:
            self.on_renamed(self.model_path.get())
        self.status.set(f"Catégorie « {name} » supprimée (fusionnée dans « {other_name} »).")


class AutomationTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.manager = AutomationManager(on_event=self._on_event)
        self._build()
        self.manager.load()
        self._schedule_refresh()

    def _build(self) -> None:
        buttons = ttk.Frame(self)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Ajouter...", command=self._add).pack(side="left")
        ttk.Button(buttons, text="Modifier...", command=self._edit).pack(side="left", padx=6)
        ttk.Button(buttons, text="Supprimer", command=self._delete).pack(side="left")
        ttk.Button(buttons, text="Démarrer", command=self._start_selected).pack(side="left", padx=(20, 6))
        ttk.Button(buttons, text="Arrêter", command=self._stop_selected).pack(side="left")

        columns = ("name", "watch", "model", "interval", "status", "last_run", "count")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=8)
        headers = {
            "name": "Nom", "watch": "Dossier surveillé", "model": "Modèle", "interval": "Intervalle",
            "status": "Statut", "last_run": "Dernier passage", "count": "Fichiers",
        }
        widths = {"name": 120, "watch": 220, "model": 150, "interval": 90, "status": 70, "last_run": 150, "count": 65}
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths[col])
        self.tree.pack(fill="x", pady=10)

        ttk.Label(self, text="Journal :").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(self, height=14, state="disabled")
        self.log.pack(fill="both", expand=True)

    def _on_event(self, message: str) -> None:
        self.after(0, self._log_line, message)

    def _log_line(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _refresh_tree(self) -> None:
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for job in self.manager.jobs.values():
            c = job.config
            status = "Actif" if job.running else "Arrêté"
            self.tree.insert(
                "", "end", iid=c.name,
                values=(
                    c.name, c.watch_dir, os.path.basename(c.model_path),
                    f"{c.interval_value} {c.interval_unit}", status,
                    job.last_run or "—", job.last_run_count,
                ),
            )
        for item_id in selected:
            if self.tree.exists(item_id):
                self.tree.selection_add(item_id)

    def _schedule_refresh(self) -> None:
        self._refresh_tree()
        self.after(3000, self._schedule_refresh)

    def _selected_name(self) -> str | None:
        selection = self.tree.selection()
        return selection[0] if selection else None

    def _add(self) -> None:
        AutomationDialog(self, on_submit=self._on_created)

    def _on_created(self, config: AutomationConfig, replacing: bool) -> None:
        if replacing and config.name in self.manager.jobs:
            self.manager.jobs.pop(config.name).stop()
        job = self.manager.add(config)
        try:
            job.start()
        except Exception as exc:
            self.manager.on_event(f"[{config.name}] Impossible de démarrer : {exc}")
        self._refresh_tree()

    def _edit(self) -> None:
        name = self._selected_name()
        if not name:
            return
        AutomationDialog(self, on_submit=self._on_created, existing=self.manager.jobs[name].config)

    def _delete(self) -> None:
        name = self._selected_name()
        if not name:
            return
        if messagebox.askyesno("Confirmer", f"Supprimer l'automatisation {name!r} ?"):
            self.manager.remove(name)
            self._refresh_tree()

    def _start_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        job = self.manager.jobs[name]
        job.config.enabled = True
        self.manager.save()
        try:
            job.start()
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
        self._refresh_tree()

    def _stop_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        job = self.manager.jobs[name]
        job.config.enabled = False
        self.manager.save()
        job.stop()
        self._refresh_tree()

    def shutdown(self) -> None:
        self.manager.stop_all()


# ── Onglet API ───────────────────────────────────────────────────
class ApiTab(ttk.Frame):
    """Démarre/arrête le serveur API local et documente ses routes, pour
    qu'un autre programme puisse piloter l'application (entraînement,
    classification, transformation des catégories, historique) sans
    réimplémenter sa logique. N'écoute que sur 127.0.0.1 ; toutes les routes
    sauf "/" et "/health" exigent la clé affichée ici, régénérée à chaque
    démarrage du serveur."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.server: api_server.ApiServer | None = None
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Port :").pack(side="left")
        self.port_var = tk.IntVar(value=get_config().api_port)
        ttk.Spinbox(top, from_=1024, to=65535, textvariable=self.port_var, width=8).pack(side="left", padx=6)
        self.toggle_button = ttk.Button(top, text="Démarrer le serveur API", command=self._toggle_server)
        self.toggle_button.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Serveur arrêté.")
        ttk.Label(self, textvariable=self.status_var, foreground="#555").pack(anchor="w", pady=(8, 0))

        key_row = ttk.Frame(self)
        key_row.pack(fill="x", pady=(4, 0))
        ttk.Label(key_row, text="Clé API (en-tête \"Authorization: Bearer <clé>\") :").pack(side="left")
        self.token_var = tk.StringVar(value="(démarrez le serveur pour générer une clé)")
        ttk.Entry(key_row, textvariable=self.token_var, width=38, state="readonly").pack(side="left", padx=6)
        ttk.Button(key_row, text="Copier", command=self._copy_token).pack(side="left")

        warning = tk.Label(
            self,
            text="N'écoute que sur cette machine (127.0.0.1), jamais sur le réseau. "
            "Tout programme lancé sur cette machine et connaissant la clé peut piloter l'application : "
            "ne la partagez pas, et arrêtez le serveur quand vous n'en avez plus besoin.",
            justify="left", wraplength=760, anchor="w", bg="#fff3cd", fg="#664d03", padx=8, pady=6,
        )
        warning.pack(fill="x", pady=(8, 0))

        ttk.Label(self, text="Routes disponibles :", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(14, 4))
        self.doc_text = scrolledtext.ScrolledText(self, height=14, state="disabled", wrap="word")
        self.doc_text.pack(fill="both", expand=True)
        self._fill_docs()

        ttk.Label(self, text="Journal des requêtes :").pack(anchor="w", pady=(10, 2))
        self.log = scrolledtext.ScrolledText(self, height=8, state="disabled")
        self.log.pack(fill="both", expand=True)

    def _fill_docs(self) -> None:
        self.doc_text.configure(state="normal")
        self.doc_text.delete("1.0", "end")
        for method, path, description in api_server.ROUTES:
            self.doc_text.insert("end", f"{method:5s} {path}\n      {description}\n\n")
        self.doc_text.configure(state="disabled")

    def _toggle_server(self) -> None:
        if self.server and self.server.running:
            self.server.stop()
            self.toggle_button.configure(text="Démarrer le serveur API")
            self.status_var.set("Serveur arrêté.")
            self.token_var.set("(démarrez le serveur pour générer une clé)")
            return

        port = self.port_var.get()
        self.server = api_server.ApiServer(port, on_event=self._log_event)
        try:
            self.server.start()
        except OSError as exc:
            messagebox.showerror("Erreur", f"Impossible de démarrer le serveur sur le port {port} :\n{exc}")
            self.server = None
            return
        self.toggle_button.configure(text="Arrêter le serveur API")
        self.status_var.set(f"Serveur actif sur http://127.0.0.1:{port}/")
        self.token_var.set(self.server.token)

    def _copy_token(self) -> None:
        if not self.server:
            return
        self.clipboard_clear()
        self.clipboard_append(self.server.token)

    def _log_event(self, message: str) -> None:
        self.after(0, self._log_line, message)

    def _log_line(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def shutdown(self) -> None:
        if self.server and self.server.running:
            self.server.stop()


# ── Onglet Paramètres ────────────────────────────────────────────
class SettingsTab(ttk.Frame):
    """Paramètres techniques de l'application, enregistrés dans
    config.json. "Enregistrer" les applique immédiatement aux opérations
    suivantes (entraînement, classification, automatisation...), sans avoir
    à redémarrer l'application. Les réglages bas niveau qui pourraient
    casser l'extraction ou la vectorisation si mal réglés restent dans le
    code plutôt que d'être exposés ici."""

    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.vars: dict[str, tk.Variable] = {}
        self._build()
        self._load_from_config()

    def _build(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        clustering = ttk.LabelFrame(container, text="Regroupement automatique (Entraînement)", padding=10)
        clustering.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_int_field(clustering, 0, "cluster_k_min", "Nombre minimal de catégories à essayer :")
        self._add_int_field(clustering, 1, "cluster_k_max", "Nombre maximal de catégories à essayer :")
        self._add_int_field(clustering, 2, "cluster_naming_top_words", "Mots-clés utilisés pour nommer une catégorie :")
        self._add_float_field(
            clustering, 3, "cluster_min_silhouette",
            "Score de silhouette minimal pour découper (sinon 1 seule catégorie) :",
        )
        self._add_int_field(
            clustering, 4, "cluster_min_cluster_size",
            "Documents minimum par catégorie pour qu'un découpage soit retenu :",
        )
        ttk.Label(
            clustering,
            text="Aucune autre limite n'est appliquée dans le code : augmentez le maximum "
            "librement si vous avez besoin de plus de catégories. Si le meilleur découpage "
            "trouvé reste sous le score de silhouette minimal, ou ne respecte pas le nombre "
            "minimum de documents par catégorie, aucune catégorie n'est forcée : le modèle "
            "garde tous les documents dans une seule catégorie (utile pour un dossier qui ne "
            "contient en réalité qu'un seul type de document).",
            foreground="#555", wraplength=320, justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        vectorization = ttk.LabelFrame(container, text="Vectorisation", padding=10)
        vectorization.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self._add_int_field(vectorization, 0, "tfidf_max_features", "Vocabulaire TF-IDF maximal :")
        self._add_int_field(vectorization, 1, "tfidf_ngram_max", "Taille maximale des groupes de mots (n-grammes) :")
        self._add_combo_field(
            vectorization, 2, "embedding_model_default", "Modèle d'embeddings par défaut :",
            [name for name, _description in EMBEDDING_MODEL_CATALOG],
        )

        classification = ttk.LabelFrame(container, text="Classification", padding=10)
        classification.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_float_field(classification, 0, "confidence_threshold", "Seuil de confiance minimal (0 à 1) :")
        self._add_str_field(classification, 1, "uncertain_category_name", 'Nom de la catégorie "incertain" :')
        self._add_str_field(classification, 2, "unreadable_category_name", 'Nom de la catégorie "illisible" :')
        self._add_str_field(classification, 3, "other_category_name", 'Nom de la catégorie "autre" (suppression) :')
        self._add_bool_field(
            classification, 4,
            "classification_improve_model_default",
            'Case "Améliorer le modèle" cochée par défaut (Classification)',
        )
        self._add_bool_field(
            classification, 5,
            "classification_export_uncertain_default",
            'Case "Inclure les fichiers à vérifier dans l\'export" cochée par défaut',
        )
        self._add_bool_field(
            classification, 6,
            "classification_export_unreadable_default",
            'Case "Inclure les fichiers non catégorisé dans l\'export" cochée par défaut',
        )

        folders = ttk.LabelFrame(container, text="Dossiers de sortie", padding=10)
        folders.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        self._add_str_field(folders, 0, "models_root", "Dossier racine des modèles (storage/models/<nom>/) :")
        self._add_str_field(folders, 1, "default_output_dir", "Dossier de sortie par défaut (Classification) :")

        automation = ttk.LabelFrame(container, text="Automatisation", padding=10)
        automation.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_str_field(automation, 0, "automation_config_path", "Fichier de configuration des automatisations :")
        self._add_int_field(automation, 1, "automation_default_interval_value", "Intervalle par défaut (valeur) :")
        self._add_combo_field(
            automation, 2, "automation_default_interval_unit", "Intervalle par défaut (unité) :",
            ["minutes", "heures", "jours"],
        )
        self._add_bool_field(automation, 3, "automation_default_move", "Déplacer les fichiers par défaut")
        self._add_bool_field(
            automation, 4,
            "automation_default_include_uncertain",
            'Inclure les fichiers "à vérifier" par défaut (nouvelle automatisation)',
        )
        self._add_bool_field(
            automation, 5,
            "automation_default_include_unreadable",
            'Inclure les fichiers "non catégorisé" par défaut (nouvelle automatisation)',
        )

        misc = ttk.LabelFrame(container, text="Divers", padding=10)
        misc.grid(row=2, column=1, sticky="nsew", pady=(0, 10))
        self._add_int_field(misc, 0, "model_discovery_max_depth", "Profondeur de recherche des modèles .pkl :")
        self._add_int_field(misc, 1, "model_history_keep", "Instantanés conservés par modèle (0 = désactivé) :")
        self._add_int_field(misc, 2, "window_width", "Largeur de fenêtre par défaut :")
        self._add_int_field(misc, 3, "window_height", "Hauteur de fenêtre par défaut :")
        self._add_int_field(misc, 4, "api_port", "Port du serveur API local (onglet API) :")

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Enregistrer", command=self._save).pack(side="left")
        ttk.Button(actions, text="Réinitialiser aux valeurs par défaut", command=self._reset_defaults).pack(
            side="left", padx=6
        )

        self.status = tk.StringVar(value=f"Fichier de configuration : {os.path.abspath(DEFAULT_CONFIG_PATH)}")
        ttk.Label(self, textvariable=self.status, foreground="#555").pack(fill="x", pady=(8, 0))

    def _add_int_field(self, parent: ttk.Frame, row: int, field_name: str, label: str) -> None:
        ttk.Label(parent, text=label, wraplength=260, justify="left").grid(row=row, column=0, sticky="w", pady=3)
        var = tk.IntVar()
        ttk.Spinbox(parent, from_=0, to=1_000_000, textvariable=var, width=10).grid(
            row=row, column=1, sticky="e", padx=(10, 0)
        )
        self.vars[field_name] = var

    def _add_float_field(self, parent: ttk.Frame, row: int, field_name: str, label: str) -> None:
        ttk.Label(parent, text=label, wraplength=260, justify="left").grid(row=row, column=0, sticky="w", pady=3)
        var = tk.DoubleVar()
        ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.05, textvariable=var, width=10).grid(
            row=row, column=1, sticky="e", padx=(10, 0)
        )
        self.vars[field_name] = var

    def _add_str_field(self, parent: ttk.Frame, row: int, field_name: str, label: str) -> None:
        ttk.Label(parent, text=label, wraplength=260, justify="left").grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var, width=24).grid(row=row, column=1, sticky="e", padx=(10, 0))
        self.vars[field_name] = var

    def _add_combo_field(self, parent: ttk.Frame, row: int, field_name: str, label: str, values: list[str]) -> None:
        ttk.Label(parent, text=label, wraplength=260, justify="left").grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar()
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=22).grid(
            row=row, column=1, sticky="e", padx=(10, 0)
        )
        self.vars[field_name] = var

    def _add_bool_field(self, parent: ttk.Frame, row: int, field_name: str, label: str) -> None:
        var = tk.BooleanVar()
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        self.vars[field_name] = var

    def _load_from_config(self, config: AppConfig | None = None) -> None:
        config = config or get_config()
        for field_name, var in self.vars.items():
            var.set(getattr(config, field_name))

    def _save(self) -> None:
        try:
            values = {name: var.get() for name, var in self.vars.items()}
            new_config = AppConfig(**values)
        except Exception as exc:
            messagebox.showerror("Erreur", f"Valeur invalide : {exc}")
            return
        save_config(new_config)
        reload_config()
        self.status.set(f"Enregistré dans {os.path.abspath(DEFAULT_CONFIG_PATH)} — appliqué immédiatement.")
        messagebox.showinfo("Terminé", "Paramètres enregistrés et appliqués.")

    def _reset_defaults(self) -> None:
        self._load_from_config(AppConfig())


# ── Fenêtre principale ────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Misc):
        self.root = root
        self.root.title("Classeur de documents")
        config = get_config()
        self.root.geometry(f"{config.window_width}x{config.window_height}")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        # Classification reste toujours le premier onglet (à gauche) ; le
        # reste du flux de travail (entraîner → renommer → automatiser) est
        # regroupé à sa droite, dans l'ordre où on s'en sert.
        self.classify_tab = ClassifyTab(notebook, on_model_improved=self._on_model_trained)
        self.transform_tab = TransformTab(notebook, on_renamed=self._on_categories_renamed)
        self.train_tab = TrainTab(notebook, on_model_created=self._on_model_trained)
        self.automation_tab = AutomationTab(notebook)
        self.settings_tab = SettingsTab(notebook)
        self.api_tab = ApiTab(notebook)

        # ttk.Notebook n'a pas de marge native entre onglets : un onglet
        # vide et désactivé (non cliquable) sert d'espaceur visuel.
        spacer_left = ttk.Frame(notebook)
        spacer_mid = ttk.Frame(notebook)
        spacer_right = ttk.Frame(notebook)
        spacer_api = ttk.Frame(notebook)
        SPACER_LABEL = " " * 8

        notebook.add(self.classify_tab, text="Classification")
        notebook.add(spacer_left, text=SPACER_LABEL)
        notebook.add(self.train_tab, text="Entraînement")
        notebook.add(self.transform_tab, text="Transformer les données")
        notebook.add(spacer_mid, text=SPACER_LABEL)
        notebook.add(self.automation_tab, text="Automatisation")
        notebook.add(spacer_right, text=SPACER_LABEL)
        notebook.add(self.settings_tab, text="Paramètres")
        notebook.add(spacer_api, text=SPACER_LABEL)
        notebook.add(self.api_tab, text="API")

        notebook.tab(spacer_left, state="disabled")
        notebook.tab(spacer_mid, state="disabled")
        notebook.tab(spacer_right, state="disabled")
        notebook.tab(spacer_api, state="disabled")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_model_trained(self, model_path: str) -> None:
        self.classify_tab._refresh_model_picker()
        self.transform_tab._refresh_model_picker()
        self.transform_tab._load_model(model_path)

    def _on_categories_renamed(self, model_path: str) -> None:
        self.classify_tab.reload_if_active(model_path)

    def _on_close(self) -> None:
        self.automation_tab.shutdown()
        self.api_tab.shutdown()
        self.root.destroy()


def main() -> None:
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
