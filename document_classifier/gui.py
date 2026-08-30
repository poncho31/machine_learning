"""Interface graphique complète : classification, entraînement et
automatisation, dans une même fenêtre à onglets.
"""
from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

from . import api_server, model_store
from .automation import AutomationConfig, AutomationManager
from .classify import (
    confirmed_override,
    known_categories,
    load_model_for_prediction,
    model_extensions,
    predict_labels,
    uncertain_category,
    unreadable_category,
)
from .config import DEFAULT_CONFIG_PATH, AppConfig, get_config, reload_config, save_config
from .discover import build_model as build_model_fn
from .discover import delete_training_duplicates as delete_training_duplicates_fn
from .discover import detected_category_for_document
from .discover import improve_model as improve_model_fn
from .extraction import ExtractedDocument, extract_text, list_documents
from .features import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_IMAGE_MODEL,
    EMBEDDING_MODEL_CATALOG,
    ENGINE_EMBEDDINGS,
    ENGINE_IMAGE,
    ENGINE_TFIDF,
    IMAGE_MODEL_CATALOG,
)
from .formats import SUPPORTED_EXTENSIONS
from .rename import (
    add_files_to_category,
    compute_dataset_vectors as compute_dataset_vectors_fn,
    delete_category_permanently,
    delete_file_from_category,
    list_category_files,
    move_files_to_category,
    rename_categories,
    rename_files_with_prefix,
)
from .utils import (
    detect_duplicate_pairs,
    dispatch_file,
    duplicate_removal_candidates,
    move_files_to_local_backup,
    write_json_atomic,
)

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
        try:
            self.size_bytes: int | None = os.path.getsize(path)
        except OSError:
            self.size_bytes = None
        self.text = ""
        # Message d'erreur d'extraction (voir extraction.extract_text), si
        # le texte n'a pas pu être lu — auparavant silencieusement ignoré,
        # ce qui ne laissait aucun moyen de savoir POURQUOI un fichier
        # ressortait vide (paquet optionnel manquant, fichier corrompu...).
        self.extraction_error: str | None = None
        self.predicted_category = ""
        # Nom calculé à partir des mots-clés dominants DU DOCUMENT LUI-MÊME
        # (voir discover.detected_category_for_document) — distinct de
        # `predicted_category` (qui applique confirmed_overrides puis
        # cherche le cluster K-Means le plus proche) : donne un aperçu
        # concret du sujet du document, y compris quand la prédiction est
        # incertaine ("a_verifier") et n'aide pas à décider d'une correction.
        self.detected_category = ""
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
        super().__init__(parent)
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self.body = scrollable.body

        self.on_model_improved = on_model_improved
        self.bundle = None
        self.engine = None
        self.model_path = tk.StringVar(value="(aucun modèle chargé)")
        self.output_dir = tk.StringVar(value=os.path.abspath(get_config().default_output_dir))
        self.move_files = tk.BooleanVar(value=False)
        self.improve_model_var = tk.BooleanVar(value=get_config().classification_improve_model_default)
        self.improve_only_var = tk.BooleanVar(value=False)
        self.export_uncertain_var = tk.BooleanVar(value=get_config().classification_export_uncertain_default)
        self.export_unreadable_var = tk.BooleanVar(value=get_config().classification_export_unreadable_default)
        self.recursive_var = tk.BooleanVar(value=True)
        self.rows: dict[str, Row] = {}
        self.discovered_models: list[tuple[str, int]] = []
        self.selected_row: Row | None = None
        self.last_dispatch_dir: str | None = None
        self._duplicate_pairs: list[dict] = []

        self._build_layout()
        self._refresh_model_picker()
        self._preselect_last_model()

    def _preselect_last_model(self) -> None:
        """Recharge automatiquement le dernier modèle utilisé (mémorisé par
        `_remember_last_model`), pour ne pas avoir à le rechoisir à chaque
        lancement de l'application."""
        last_path = get_config().last_model_path
        if not last_path or not os.path.exists(last_path):
            return
        for index, (path, _size) in enumerate(self.discovered_models):
            if os.path.abspath(path) == os.path.abspath(last_path):
                self.model_picker.current(index)
                break
        self._load_model(last_path)

    def _build_layout(self) -> None:
        ttk.Label(self.body, text="1. Modèle à utiliser :").pack(anchor="w")

        top = ttk.Frame(self.body)
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
        self.history_button = ttk.Button(
            browse_row, text="Historique / Revenir en arrière...", command=self._open_history_dialog, state="disabled"
        )
        self.history_button.pack(side="left", padx=6)
        ttk.Label(browse_row, textvariable=self.model_path, foreground="#555").pack(side="left", padx=8)

        ttk.Label(self.body, text="2. Documents à classer :").pack(anchor="w", pady=(16, 6))

        buttons = ttk.Frame(self.body)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Ajouter des fichiers...", command=self._add_files_dialog).pack(side="left")
        ttk.Button(buttons, text="Ajouter un dossier...", command=self._add_folder_dialog).pack(side="left", padx=6)
        ttk.Checkbutton(buttons, text="Inclure les sous-dossiers", variable=self.recursive_var).pack(
            side="left", padx=(0, 12)
        )
        ttk.Button(buttons, text="Vider la liste", command=self._clear_rows).pack(side="left")
        ttk.Button(buttons, text="Détecter les doublons", command=self._detect_duplicates).pack(
            side="left", padx=(18, 0)
        )
        self.delete_duplicates_button = ttk.Button(
            buttons, text="Supprimer les doublons", command=self._delete_duplicates, state="disabled",
        )
        self.delete_duplicates_button.pack(side="left", padx=6)
        ttk.Button(
            buttons, text="Attribuer une catégorie...", command=self._edit_selected_category,
        ).pack(side="left", padx=(18, 0))

        paned = ttk.PanedWindow(self.body, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=10)

        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        columns = ("filename", "size", "predicted", "confidence", "detected_category", "corrected")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("filename", text="Fichier")
        self.tree.heading("size", text="Taille")
        self.tree.heading("predicted", text="Catégorie proposée")
        self.tree.heading("confidence", text="Confiance")
        self.tree.heading("detected_category", text="Catégorie détectée (mots-clés)")
        self.tree.heading("corrected", text="Catégorie retenue")
        self.tree.column("filename", width=320)
        self.tree.column("size", width=80, anchor="e")
        self.tree.column("predicted", width=180)
        self.tree.column("confidence", width=90, anchor="center")
        self.tree.column("detected_category", width=200)
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
        preview_buttons = ttk.Frame(preview_frame)
        preview_buttons.pack(anchor="w")
        self.open_file_button = ttk.Button(
            preview_buttons, text="Ouvrir le fichier original", command=self._open_selected_file, state="disabled"
        )
        self.open_file_button.pack(side="left")
        self.open_text_button = ttk.Button(
            preview_buttons, text="Ouvrir le fichier texte", command=self._open_selected_text, state="disabled",
        )
        self.open_text_button.pack(side="left", padx=(6, 0))

        ttk.Label(self.body, text="3. Validation et export :").pack(anchor="w", pady=(10, 6))

        bottom = ttk.Frame(self.body)
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

        ttk.Checkbutton(
            bottom,
            text="Ne pas exporter les fichiers — juste améliorer le modèle (aucun dossier créé)",
            variable=self.improve_only_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.improve_progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.improve_progress.grid(row=6, column=0, columnspan=3, sticky="we", pady=(6, 0))

        self.status = tk.StringVar(value="Chargez un modèle pour commencer.")
        ttk.Label(self.body, textvariable=self.status, foreground="#555").pack(fill="x", pady=(4, 0))

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
        self.history_button.configure(state="normal")
        self._remember_last_model(path)

    def _remember_last_model(self, path: str) -> None:
        """Retient le modèle chargé pour le présélectionner automatiquement
        au prochain lancement de l'application (voir __init__)."""
        config = get_config()
        absolute_path = os.path.abspath(path)
        if config.last_model_path == absolute_path:
            return
        config.last_model_path = absolute_path
        try:
            save_config(config)
        except OSError:
            pass  # non bloquant : au pire, rien à présélectionner la prochaine fois

    # ── Ajout de fichiers ──
    def _model_extensions(self) -> tuple[str, ...]:
        """Types de fichiers du modèle chargé (voir `classify.model_extensions`) :
        un modèle entraîné uniquement sur des `.pdf` ne doit faire remonter
        que des `.pdf` quand on parcourt un dossier. Tous les formats pris en
        charge si aucun modèle n'est encore chargé."""
        return model_extensions(self.bundle) if self.bundle is not None else SUPPORTED_EXTENSIONS

    # ── Historique ──
    def _open_history_dialog(self) -> None:
        model_path = self.model_path.get()
        if not model_path or not os.path.exists(model_path):
            return
        HistoryDialog(self, model_path, on_restored=self._on_history_restored)

    def _on_history_restored(self, model_path: str) -> None:
        self._load_model(model_path)

    # ── Ajout de fichiers ──
    def _add_files_dialog(self) -> None:
        paths = filedialog.askopenfilenames(title="Choisir des documents", filetypes=FILETYPES_DOCS)
        self._add_paths(paths)

    def _add_folder_dialog(self) -> None:
        directory = filedialog.askdirectory(title="Choisir un dossier de documents")
        if not directory:
            return
        self._add_paths(
            list_documents(directory, recursive=self.recursive_var.get(), extensions=self._model_extensions())
        )

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
            try:
                text, error = extract_text(path)
                row.text = text
                row.extraction_error = error
                if not text.strip():
                    row.predicted_category = unreadable_category()
                    row.confidence = 0.0
                    # Le fichier n'a pas pu être lu : ce n'est pas une prédiction
                    # incertaine, donc pas un cas "a_verifier" — le seuil de
                    # confiance ne s'applique pas ici.
                    row.corrected_category = unreadable_category()
                else:
                    # Un document dont le contenu correspond exactement à une
                    # correction déjà confirmée à la main (onglet Classification,
                    # "Améliorer le modèle avec ces documents") doit retrouver
                    # cette catégorie directement ici, pas seulement lors d'un
                    # export via `classify()` (voir `classify.confirmed_override`)
                    # — sinon ce même document réapparaît sous la prédiction brute
                    # du clustering dès qu'on le rajoute à la liste, comme s'il
                    # n'avait jamais été confirmé.
                    override = confirmed_override(self.bundle, text)
                    if override is not None:
                        row.predicted_category = override
                        row.confidence = 1.0
                    else:
                        vectors = self.engine.transform([text])
                        labels, confidences = predict_labels(self.bundle, vectors)
                        row.predicted_category = labels[0]
                        row.confidence = float(confidences[0])
                    # Calculé à partir des mots-clés dominants DU DOCUMENT
                    # LUI-MÊME (voir Row.detected_category ci-dessus) : utile
                    # même quand `predicted_category` retombe sur "a_verifier",
                    # pour donner un indice concret du sujet du document plutôt
                    # que de laisser l'utilisateur deviner sans rien.
                    row.detected_category = detected_category_for_document(self.bundle, self.engine, text) or ""
                    row.corrected_category = (
                        row.predicted_category if row.confidence >= threshold else uncertain_category()
                    )
            except Exception as exc:
                # Une erreur inattendue sur CE fichier ne doit jamais
                # interrompre le traitement du reste du lot en silence : sans
                # ce filet, une exception ici tuait le thread entier — les
                # fichiers suivants n'apparaissaient tout simplement jamais
                # dans le tableau, sans aucun message (ex. vécu : une erreur
                # "empty vocabulary" de scikit-learn sur un texte trop court
                # coupait net la liste après ce fichier).
                row.text = ""
                row.extraction_error = str(exc)
                row.predicted_category = unreadable_category()
                row.confidence = 0.0
                row.corrected_category = unreadable_category()
            self.after(0, self._insert_row, row)
        self.after(0, lambda: self.status.set(f"{len(self.rows)} fichier(s) au total."))

    def _insert_row(self, row: Row) -> None:
        confidence_display = f"{row.confidence:.0%}" if row.confidence is not None else "n/a"
        size_display = _human_size(row.size_bytes) if row.size_bytes is not None else "n/a"
        tag = ()
        if row.predicted_category == unreadable_category():
            tag = ("unreadable",)
        elif row.confidence is not None and row.confidence < get_config().confidence_threshold:
            tag = ("uncertain",)
        row.item_id = self.tree.insert(
            "", "end",
            values=(
                row.filename, size_display, row.predicted_category, confidence_display,
                row.detected_category or "n/a", row.corrected_category,
            ),
            tags=tag,
        )

    def _clear_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.selected_row = None
        self._set_preview_text("")
        self.open_file_button.configure(state="disabled")
        self._duplicate_pairs = []
        self.delete_duplicates_button.configure(state="disabled")
        self.status.set("Liste vidée.")

    # ── Doublons ──
    def _detect_duplicates(self) -> None:
        readable_rows = [r for r in self.rows.values() if r.text.strip()]
        if len(readable_rows) < 2:
            messagebox.showinfo(
                "Détection des doublons", "Il faut au moins 2 fichiers lisibles dans la liste pour comparer."
            )
            return
        if self.engine is None:
            messagebox.showwarning("Aucun modèle", "Chargez un modèle avant de détecter les doublons.")
            return

        config = get_config()
        vectors = self.engine.transform([r.text for r in readable_rows])
        pairs = detect_duplicate_pairs(
            [r.path for r in readable_rows], [r.filename for r in readable_rows], vectors,
            threshold=config.cluster_duplicate_threshold, max_docs=config.cluster_duplicate_max_docs,
        )
        self._duplicate_pairs = pairs

        if not pairs:
            self.delete_duplicates_button.configure(state="disabled")
            messagebox.showinfo(
                "Détection des doublons", f"Aucun doublon détecté parmi {len(readable_rows)} fichier(s)."
            )
            return

        self.delete_duplicates_button.configure(state="normal")
        DuplicatesDialog(self, pairs, on_confirm=lambda: self._delete_duplicates(confirm=False))

    def _delete_duplicates(self, confirm: bool = True) -> None:
        if not self._duplicate_pairs:
            return
        to_remove = duplicate_removal_candidates(self._duplicate_pairs)
        if not to_remove:
            return
        if confirm and not messagebox.askyesno(
            "Confirmer",
            f"Déplacer {len(to_remove)} document(s) en double vers un dossier « _backup » à côté de "
            "leur dossier d'origine ? Ce n'est jamais une suppression définitive.",
        ):
            return

        moved = move_files_to_local_backup(to_remove)
        for path in to_remove:
            row = self.rows.pop(path, None)
            if row and row.item_id:
                self.tree.delete(row.item_id)

        self._duplicate_pairs = []
        self.delete_duplicates_button.configure(state="disabled")
        self.status.set(f"{len(moved)} document(s) en double déplacé(s) vers un dossier « _backup ».")
        messagebox.showinfo(
            "Doublons déplacés",
            f"{len(moved)} document(s) déplacé(s) vers un dossier « _backup » à côté de leur dossier d'origine.",
        )

    # ── Aperçu du fichier sélectionné ──
    def _on_select_row(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            self.selected_row = None
            self._set_preview_text("")
            self.open_file_button.configure(state="disabled")
            self.open_text_button.configure(state="disabled")
            return
        if len(selection) > 1:
            # Plusieurs fichiers sélectionnés à la fois (voir _edit_selected_category
            # pour leur attribuer la même catégorie en un geste) : l'aperçu et
            # "Ouvrir le fichier original"/"Ouvrir le fichier texte" ne
            # s'appliquent qu'à UN fichier, donc désactivés plutôt que de
            # montrer arbitrairement le premier.
            self.selected_row = None
            self._set_preview_text(f"({len(selection)} fichiers sélectionnés)")
            self.open_file_button.configure(state="disabled")
            self.open_text_button.configure(state="disabled")
            return
        row = next((r for r in self.rows.values() if r.item_id == selection[0]), None)
        self.selected_row = row
        if row is None:
            return
        if row.text.strip():
            preview = row.text
        elif row.extraction_error:
            # Montrer la VRAIE raison (paquet optionnel manquant, fichier
            # corrompu/chiffré...) plutôt qu'un message générique qui ne
            # permet pas de savoir quoi corriger.
            preview = f"(aucun texte n'a pu être extrait de ce fichier)\n\nDétail de l'erreur :\n{row.extraction_error}"
        else:
            preview = "(aucun texte n'a pu être extrait de ce fichier)"
        self._set_preview_text(preview)
        self.open_file_button.configure(state="normal")
        self.open_text_button.configure(state="normal" if row.text.strip() else "disabled")

    def _set_preview_text(self, text: str) -> None:
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def _open_selected_file(self) -> None:
        if self.selected_row and os.path.isfile(self.selected_row.path):
            os.startfile(self.selected_row.path)

    def _open_selected_text(self) -> None:
        """Ouvre le texte EXTRAIT (pas le fichier d'origine) dans l'éditeur
        de texte par défaut de Windows — utile pour lire le contenu complet
        d'un format que "Ouvrir le fichier original" n'ouvre pas dans une
        application lisible telle quelle (.msg sans Outlook installé, par
        exemple), ou simplement pour le copier ailleurs sans se limiter à la
        fenêtre d'aperçu. Écrit dans un dossier temporaire dédié, réutilisé
        (écrasé) à chaque réouverture du même fichier plutôt que d'accumuler
        une copie par clic."""
        row = self.selected_row
        if row is None or not row.text.strip():
            return
        preview_dir = os.path.join(tempfile.gettempdir(), "classeur_documents_apercu")
        try:
            os.makedirs(preview_dir, exist_ok=True)
            text_path = os.path.join(preview_dir, f"{row.filename}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(row.text)
        except OSError as exc:
            messagebox.showerror("Erreur", f"Impossible d'écrire le fichier texte temporaire :\n{exc}")
            return
        os.startfile(text_path)

    # ── Correction manuelle ──
    def _edit_selected_category(self, _event=None) -> None:
        """Double-clic sur une ligne, ou bouton "Attribuer une catégorie..."
        (nécessaire pour une sélection multiple : un simple clic sur une
        ligne, y compris le premier clic d'un double-clic, réduit toujours la
        sélection du Treeview à cette seule ligne — le bouton, lui, agit sur
        la sélection Ctrl/Shift+clic déjà construite, sans la perturber).
        Attribue la même catégorie à TOUS les fichiers sélectionnés en une
        seule fois, plutôt que de devoir corriger chaque fichier un par un."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Aucune sélection", "Sélectionnez d'abord un ou plusieurs fichiers dans la liste.")
            return
        rows = [r for r in self.rows.values() if r.item_id in selection]
        if not rows:
            return

        categories = known_categories(self.bundle) if self.bundle else []
        categories = sorted(set(categories) | {r.predicted_category for r in rows} | {uncertain_category()})

        # Valeur de départ du champ : la catégorie retenue commune à tous les
        # fichiers sélectionnés si elle est unique, vide sinon (pour ne pas
        # laisser croire que celle du premier fichier s'appliquerait à tous).
        corrected_values = {r.corrected_category for r in rows}
        initial_value = corrected_values.pop() if len(corrected_values) == 1 else ""

        popup = tk.Toplevel(self)
        title = f"Catégorie — {rows[0].filename}" if len(rows) == 1 else f"Catégorie — {len(rows)} fichiers sélectionnés"
        popup.title(title)
        popup.geometry("340x100")
        ttk.Label(popup, text="Catégorie retenue :").pack(padx=10, pady=(10, 2), anchor="w")
        var = tk.StringVar(value=initial_value)
        combo = ttk.Combobox(popup, textvariable=var, values=categories)
        combo.pack(fill="x", padx=10)

        def confirm() -> None:
            new_category = var.get().strip()
            if new_category:
                for row in rows:
                    if new_category != row.corrected_category:
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

        # "Ne pas exporter les fichiers" saute complètement la création du
        # dossier d'export : sans "Améliorer le modèle" en plus, valider ne
        # ferait alors littéralement rien — préférable de le dire clairement
        # plutôt que de laisser l'utilisateur cliquer dans le vide.
        improve_only = self.improve_only_var.get()
        if improve_only and not self.improve_model_var.get():
            messagebox.showwarning(
                "Rien à faire",
                "« Ne pas exporter les fichiers » est coché mais pas « Améliorer le modèle » : "
                "aucune des deux actions ne serait effectuée. Cochez au moins l'une des deux.",
            )
            return

        run_dir = None
        if not improve_only:
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
        # Corrections manuelles qui NE PEUVENT PAS contribuer à l'amélioration
        # du modèle faute de texte exploitable (PDF scanné sans OCR, fichier
        # corrompu...) : l'export les classe correctement quand même (il ne
        # dépend que de la catégorie choisie, pas du texte), mais sans texte
        # il n'y a rien à vectoriser pour le ré-entraînement. Sans ce
        # compteur, "Améliorer le modèle" coché échouait silencieusement pour
        # ces fichiers-là, sans aucune indication à l'utilisateur.
        unreadable_corrections = 0

        for row in self.rows.values():
            category = row.corrected_category or uncertain_category()
            if category == uncertain_name and not include_uncertain:
                skipped_uncertain += 1
                continue
            if category == unreadable_name and not include_unreadable:
                skipped_unreadable += 1
                continue
            try:
                if improve_only:
                    # Aucun export : le fichier ORIGINAL sert directement de
                    # source pour l'amélioration, sans copie intermédiaire —
                    # jamais déplacé ni modifié (voir `_add_confirmed_documents_to_dataset`,
                    # toujours en copie).
                    dest = row.path
                    results[row.path] = {
                        "category": category,
                        "predicted_category": row.predicted_category,
                        "confidence": row.confidence,
                    }
                else:
                    dest = dispatch_file(row.path, category, run_dir, move=self.move_files.get())
                    results[row.path] = {
                        "category": category,
                        "predicted_category": row.predicted_category,
                        "confidence": row.confidence,
                        "destination": dest,
                    }
                if row.manually_corrected and category not in placeholder_categories:
                    if row.text.strip():
                        improve_batch.append(
                            ExtractedDocument(path=dest, filename=os.path.basename(dest), text=row.text)
                        )
                        confirmed_labels[dest] = category
                    else:
                        unreadable_corrections += 1
            except Exception as exc:
                errors.append(f"{row.filename} : {exc}")

        if results and not improve_only:
            write_json_atomic(results, os.path.join(run_dir, "classification.json"))
            self.last_dispatch_dir = run_dir
            self.open_dispatch_button.configure(state="normal")

        should_improve = self.improve_model_var.get() and self.bundle is not None and improve_batch
        self._clear_rows()

        if improve_only:
            message = f"{len(results)} fichier(s) pris en compte (aucun export : aucun dossier créé)."
        else:
            message = f"{len(results)} fichier(s) classé(s) dans {run_dir}."
        if skipped_uncertain:
            message += f"\n{skipped_uncertain} fichier(s) \"à vérifier\" exclu(s)."
        if skipped_unreadable:
            message += f"\n{skipped_unreadable} fichier(s) \"non catégorisé\" exclu(s)."
        if self.improve_model_var.get() and self.bundle is not None:
            if should_improve:
                message += f"\n\nAmélioration du modèle en cours avec {len(improve_batch)} document(s) corrigé(s)..."
            elif unreadable_corrections:
                message += (
                    f"\n\n⚠ {unreadable_corrections} correction(s) manuelle(s) n'ont pas pu améliorer le modèle : "
                    "aucun texte n'a pu être extrait de ces fichiers (PDF scanné sans OCR, fichier corrompu...). "
                    + ("" if improve_only else "Ils ont bien été classés dans l'export, mais ")
                    + "le modèle n'apprend rien d'un fichier sans texte."
                )
            else:
                message += "\n\nAucune correction manuelle exploitable : le modèle n'a pas été modifié."
        if errors:
            message += "\n\nErreurs :\n" + "\n".join(errors)
            messagebox.showwarning("Terminé avec erreurs", message)
        else:
            messagebox.showinfo("Terminé", message)
        self.status.set(f"{len(results)} fichier(s) traité(s).")

        if not improve_only and results and os.path.isdir(run_dir):
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
            # Le message est construit ICI, pas dans le lambda différé :
            # Python supprime la variable `exc` à la sortie du bloc `except`,
            # donc un lambda qui la referencerait directement lèverait sa
            # propre erreur ("free variable referenced before assignment")
            # une fois exécuté plus tard par self.after, masquant le VRAI
            # message d'erreur derrière un second plantage sans rapport.
            message = f"Amélioration du modèle impossible :\n{exc}"
            self.after(0, lambda: messagebox.showerror("Erreur", message))
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
        "Aucun réglage particulier : reprend les valeurs actuelles de l'onglet Paramètres (TF-IDF).",
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
        "Mélange de plusieurs types de documents (factures + contrats + fiches de paie...)",
        "TF-IDF avec un vocabulaire plus large : des types de documents différents partagent en "
        "général peu de vocabulaire exact, TF-IDF les sépare bien sans rien télécharger.",
        {
            "engine": ENGINE_TFIDF, "tfidf_max_features": 6000, "tfidf_ngram_max": 2,
            "k_max": 20, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Gros volume de documents, priorité à la vitesse",
        "TF-IDF avec un vocabulaire réduit et des mots seuls (pas de groupes de mots) : rapide "
        "sur un très grand nombre de documents, au prix d'un peu de finesse.",
        {
            "engine": ENGINE_TFIDF, "tfidf_max_features": 2000, "tfidf_ngram_max": 1,
            "k_max": 20, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Exploration permissive (tout faire ressortir, à trier ensuite)",
        "TF-IDF, seuils très bas : fait ressortir un maximum de sous-groupes, y compris "
        "incertains, à corriger ensuite dans la section \"Catégories de ce modèle\" ci-dessous.",
        {
            "engine": ENGINE_TFIDF, "k_max": 20,
            "cluster_min_silhouette": 0.02, "cluster_min_cluster_size": 1,
        },
    ),
    (
        "Texte libre, léger et rapide (anglais)",
        "Embeddings all-MiniLM-L6-v2 (~90 Mo, le plus rapide) : un premier essai en embeddings "
        "sur un corpus hétérogène en anglais, sans attendre un téléchargement ou un calcul long.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "all-MiniLM-L6-v2",
            "k_max": 15, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Texte libre, un peu plus précis (anglais)",
        "Embeddings all-MiniLM-L12-v2 (~120 Mo) : légèrement plus précis que L6, reste rapide, "
        "pour du texte libre en anglais.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "all-MiniLM-L12-v2",
            "k_max": 15, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Texte libre en français, formulations variées (contrats, courriers, rapports)",
        "Embeddings paraphrase-multilingual-MiniLM-L12-v2 (~470 Mo) : bon compromis "
        "vitesse/précision pour du texte libre réellement rédigé en français.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "k_max": 15, "cluster_min_silhouette": 0.12, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Texte libre, meilleure qualité (anglais)",
        "Embeddings all-mpnet-base-v2 (~420 Mo) : la meilleure qualité en anglais parmi les "
        "modèles proposés, plus lent.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "all-mpnet-base-v2",
            "k_max": 15, "cluster_min_silhouette": 0.15, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Texte libre multilingue, meilleure qualité (lourd)",
        "Embeddings paraphrase-multilingual-mpnet-base-v2 (~970 Mo) : la meilleure qualité "
        "multilingue disponible (français inclus), le plus lent — pour un corpus réellement "
        "hétérogène où la précision prime sur la vitesse.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
            "k_max": 15, "cluster_min_silhouette": 0.15, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Documents très gabarités, racinisation + regroupement hiérarchique",
        "TF-IDF avec racinisation (\"facture\"/\"factures\"/\"facturé\" comptent comme un seul "
        "mot) pour ne pas diluer un vocabulaire déjà très répétitif, et \"agglomerative\" "
        "(hiérarchique) plutôt que k-means : sur un petit lot très homogène (même fournisseur, "
        "même mise en page), il découpe souvent plus proprement selon la vraie structure des "
        "sous-variantes. Seuil de documents minimum relevé pour ignorer les variations isolées.",
        {
            "engine": ENGINE_TFIDF, "tfidf_use_stemming": True, "cluster_algorithm": "agglomerative",
            "cluster_use_svd": False, "k_max": 8, "cluster_min_silhouette": 0.15, "cluster_min_cluster_size": 4,
        },
    ),
    (
        "Très gros fonds documentaire (archives, plusieurs milliers de fichiers)",
        "MiniBatchKMeans (bascule de toute façon automatique au-delà du seuil de la config, ici "
        "forcé dès le départ), vocabulaire réduit, et réduction de dimension SVD pour accélérer "
        "le regroupement sur un espace TF-IDF autrement très grand — plus la détection des "
        "doublons, fréquents sur de grosses archives accumulées au fil du temps.",
        {
            "engine": ENGINE_TFIDF, "tfidf_max_features": 3000, "cluster_algorithm": "minibatch_kmeans",
            "cluster_use_svd": True, "cluster_svd_components": 150, "detect_duplicates": True, "k_max": 25,
        },
    ),
    (
        "Fonds hétérogène avec documents atypiques à isoler (HDBSCAN)",
        "HDBSCAN, seul algorithme qui ne force pas chaque document dans un cluster : les "
        "documents trop différents du reste (un type de document rare, un scan illisible "
        "malgré tout catégorisable...) sont regroupés dans une catégorie \"autre\" plutôt que "
        "rattachés arbitrairement au groupe le moins pire — utile pour un fonds qu'on ne "
        "connaît pas encore bien. Embeddings plutôt que TF-IDF : HDBSCAN se fie à des "
        "distances de densité, plus significatives dans un espace déjà dense.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "cluster_algorithm": "hdbscan", "cluster_use_svd": False, "cluster_min_cluster_size": 3,
        },
    ),
    (
        "Fusion d'archives avec beaucoup de quasi-doublons (scans répétés)",
        "Détection des doublons activée pour ne garder qu'un exemplaire de chaque document "
        "scanné/copié plusieurs fois avant même de le regrouper, avec \"agglomerative\" (plus "
        "cohérent qu'un k-means une fois les vrais doublons écartés) et une réduction SVD pour "
        "rester rapide malgré la fusion de plusieurs dossiers d'archives.",
        {
            "engine": ENGINE_TFIDF, "detect_duplicates": True, "cluster_algorithm": "agglomerative",
            "cluster_use_svd": True, "cluster_svd_components": 100, "cluster_min_cluster_size": 2,
        },
    ),
    (
        "Vocabulaire pollué par un nom d'entreprise/adresse toujours présent",
        "Racinisation activée + liste de mots à ignorer en plus des mots vides habituels : "
        "remplacez \"ma_societe, mon_adresse, mon_siret\" ci-dessous par les mots qui reviennent "
        "dans TOUS vos documents (en-tête, pied de page...) et qui sinon dominent le nommage des "
        "catégories sans rien apporter pour les distinguer entre eux.",
        {
            "engine": ENGINE_TFIDF, "tfidf_use_stemming": True,
            "cluster_extra_stopwords": "ma_societe, mon_adresse, mon_siret", "k_max": 15,
        },
    ),
    (
        "Présentation client : noms de catégories les plus naturels (KeyBERT)",
        "Embeddings de la meilleure qualité multilingue disponible, combinés à KeyBERT pour des "
        "noms de catégorie choisis par similarité sémantique plutôt que par simple assemblage "
        "de mots-clés TF-IDF — pour un résultat présentable tel quel plutôt qu'à retravailler.",
        {
            "engine": ENGINE_EMBEDDINGS, "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
            "use_keybert": True, "cluster_min_silhouette": 0.15,
        },
    ),
    (
        "Prototypage rapide, sans dépendance optionnelle (poste minimal)",
        "TF-IDF pur, sans racinisation (pas besoin de `snowballstemmer`) ni réduction SVD, "
        "vocabulaire réduit et mots isolés : le combo le plus léger et le plus rapide pour un "
        "premier essai sur un petit corpus, ou sur un poste où seule `requirements.txt` de base "
        "a été installée.",
        {
            "engine": ENGINE_TFIDF, "tfidf_use_stemming": False, "cluster_use_svd": False,
            "cluster_algorithm": "kmeans", "tfidf_max_features": 1500, "tfidf_ngram_max": 1, "k_max": 10,
        },
    ),
    (
        "Archives de scans/photos de documents (OCR)",
        "Pense-bête : activez d'abord l'OCR dans l'onglet Paramètres (et sélectionnez les "
        "formats image/PDF scannés à l'étape 1 ci-dessus) — sans quoi un PDF sans texte natif "
        "ou une photo reste \"illisible\". Une fois l'OCR activé, ce préréglage reste du TF-IDF "
        "classique (racinisation + SVD) : le texte reconnu par Tesseract est ensuite traité "
        "exactement comme n'importe quel autre texte extrait.",
        {
            "engine": ENGINE_TFIDF, "tfidf_use_stemming": True, "cluster_use_svd": True,
        },
    ),
    (
        "Contrats et avenants (vocabulaire juridique varié)",
        "Groupes de mots plus longs (jusqu'à 3 mots) pour capturer des formulations juridiques "
        "figées (\"résiliation anticipée du contrat\"), avec \"agglomerative\" + SVD : sur des "
        "contrats dont la formulation varie beaucoup d'un modèle à l'autre, un regroupement "
        "hiérarchique sur un espace réduit tend à mieux respecter les familles de documents "
        "réellement proches que k-means.",
        {
            "engine": ENGINE_TFIDF, "tfidf_ngram_max": 3, "cluster_algorithm": "agglomerative",
            "cluster_use_svd": True, "cluster_svd_components": 120, "cluster_min_cluster_size": 3,
        },
    ),
    (
        "Photos et images — contenu visuel (personnes, scènes, objets)",
        "Moteur \"image\" : catégorise par ce qui est VISIBLE sur la photo (des personnes, un "
        "paysage, un document, un intérieur...) via un modèle CLIP, sans se soucier d'un "
        "quelconque texte imprimé dessus — capacité différente et indépendante de l'OCR (qui LIT "
        "le texte d'une image, mais ne \"voit\" ni personnes ni scène). Nomme chaque catégorie par "
        "comparaison à une liste de libellés candidats modifiable ci-dessous (\"Libellés candidats\"), "
        "puisqu'il n'y a pas de texte propre au document dont extraire des mots-clés.",
        {
            "engine": ENGINE_IMAGE, "cluster_algorithm": "hdbscan", "k_max": 15,
        },
    ),
]

# Regroupement des extensions prises en charge (voir formats.py) pour la
# sélection des types de fichiers à inclure dans un entraînement — un seul
# type, plusieurs, ou tous (cases cochées par défaut).
EXTENSION_GROUPS: list[tuple[str, list[str]]] = [
    ("Documents", [".pdf", ".docx", ".rtf"]),
    ("Texte et données", [".txt", ".md", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".ini", ".cfg", ".toml"]),
    ("Balisage", [".html", ".htm", ".xml"]),
    ("Email", [".eml", ".msg"]),
    ("Bureautique", [".xlsx", ".xlsm", ".pptx"]),
    ("OpenDocument", [".odt", ".ods", ".odp"]),
    ("Images (texte : OCR requis — contenu visuel : moteur \"image\")", [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]),
]

# Cas d'usage proposés dans l'onglet Entraînement : chacun pré-remplit les
# types de fichiers et le préréglage moteur adaptés. Seuls "available: True"
# s'appuient sur ce que l'outil sait déjà faire (catégorisation par
# similarité) ; les autres nécessiteraient une capacité différente (extraction
# de champs, détection de clauses...) pas encore implémentée — sélectionner
# l'un d'eux affiche simplement une explication, sans rien changer au formulaire.
USE_CASES: list[dict] = [
    {
        "name": "Classeur de documents (par défaut)",
        "description": (
            "Trie automatiquement des documents en vrac (factures, contrats, courriers...) "
            "dans des catégories détectées automatiquement. Mode par défaut de cet onglet : "
            "tous les types de fichiers, aucun réglage supplémentaire."
        ),
        "available": True,
        "extensions": None,
        "preset": TRAINING_PRESETS[0][0],
    },
    {
        "name": "Trieur d'emails",
        "description": (
            "Classe des emails exportés (.eml, .msg) en catégories détectées automatiquement "
            "(clients, admin, urgent...), avec le même moteur de catégorisation que le mode "
            "par défaut, limité aux formats email. Pour imposer des catégories fixes (ex. "
            "spam / non-spam), renommez les catégories détectées dans la section \"Catégories de "
            "ce modèle\" ci-dessous après l'entraînement — ce mode ne fournit pas de classifieur "
            "spam pré-entraîné."
        ),
        "available": True,
        "extensions": [".eml", ".msg"],
        "preset": "Texte libre en français, formulations variées (contrats, courriers, rapports)",
    },
    {
        "name": "Factures / fiches de paie d'un même émetteur",
        "description": (
            "Un même fournisseur ou service RH produit des documents très gabariés (même mise "
            "en page, texte presque identique d'un exemplaire à l'autre) : racinisation + "
            "regroupement hiérarchique (\"agglomerative\") pour découper selon les vraies "
            "sous-variantes plutôt que sur du bruit de mise en forme."
        ),
        "available": True,
        "extensions": None,
        "preset": "Documents très gabarités, racinisation + regroupement hiérarchique",
    },
    {
        "name": "Archives volumineuses (plusieurs milliers de fichiers)",
        "description": (
            "Un dossier d'archives accumulé depuis longtemps, avec beaucoup de fichiers et "
            "probablement des doublons/scans répétés : MiniBatchKMeans + réduction de "
            "dimension (SVD) pour rester rapide, et détection des doublons activée."
        ),
        "available": True,
        "extensions": None,
        "preset": "Très gros fonds documentaire (archives, plusieurs milliers de fichiers)",
    },
    {
        "name": "Fonds documentaire inconnu, isoler les atypiques",
        "description": (
            "Vous ne savez pas encore ce que contient vraiment ce dossier : HDBSCAN ne force "
            "pas les documents trop différents du reste dans une catégorie arbitraire, il les "
            "regroupe dans \"autre\" — plus fiable qu'un k-means classique pour un premier "
            "défrichage d'un fonds mal connu."
        ),
        "available": True,
        "extensions": None,
        "preset": "Fonds hétérogène avec documents atypiques à isoler (HDBSCAN)",
    },
    {
        "name": "Fusion de plusieurs dossiers d'archives",
        "description": (
            "Vous réunissez plusieurs anciens dossiers d'archives dans un seul modèle, avec "
            "probablement le même document scanné ou exporté plusieurs fois : détection des "
            "doublons avant le regroupement pour ne garder qu'un exemplaire de chaque."
        ),
        "available": True,
        "extensions": None,
        "preset": "Fusion d'archives avec beaucoup de quasi-doublons (scans répétés)",
    },
    {
        "name": "Scans et photos de documents (OCR)",
        "description": (
            "PDF scannés sans texte natif ou photos de documents (.png, .jpg...) : nécessite "
            "d'activer l'OCR dans l'onglet Paramètres au préalable, ET d'avoir installé le "
            "moteur Tesseract séparément sur la machine (le paquet Python pytesseract, lui, "
            "est déjà installé par défaut — voir le README pour les liens d'installation de "
            "Tesseract), sans quoi ces fichiers restent \"illisibles\" comme aujourd'hui."
        ),
        "available": True,
        "extensions": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        "preset": "Archives de scans/photos de documents (OCR)",
    },
    {
        "name": "Trieur de photos (personnes, scènes, contexte visuel)",
        "description": (
            "Trie des photos par ce qu'elles MONTRENT (personnes, paysage, intérieur, "
            "événement...), pas par un texte qu'elles contiendraient — capacité différente de "
            "l'OCR ci-dessus (qui LIT le texte visible, sans rien comprendre à la scène). "
            "Utilise un modèle CLIP (paquet sentence-transformers, déjà nécessaire pour le "
            "moteur \"embeddings\" — voir requirements-embeddings.txt), téléchargé au premier "
            "usage puis mis en cache comme les autres modèles."
        ),
        "available": True,
        "extensions": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        "preset": "Photos et images — contenu visuel (personnes, scènes, objets)",
    },
    {
        "name": "Extracteur de factures (à venir)",
        "description": (
            "Extraire automatiquement les champs clés d'une facture (date, montants HT/TTC, "
            "TVA, IBAN, fournisseur) et les exporter en CSV/Excel. Nécessiterait un modèle "
            "d'extraction de champs (NER) entraîné sur des factures annotées (ex. jeux de "
            "données SROIE, CORD) — une capacité différente de la catégorisation par similarité "
            "qu'offre l'outil aujourd'hui, pas encore implémentée."
        ),
        "available": False,
    },
    {
        "name": "Analyseur de contrats (à venir)",
        "description": (
            "Détecter automatiquement les clauses clés d'un contrat (durée, résiliation, "
            "pénalités, juridiction) et alerter sur les clauses à risque. Nécessiterait un "
            "modèle de détection de clauses (type BERT finement ajusté, ex. sur le jeu de "
            "données CUAD) — pas encore implémenté."
        ),
        "available": False,
    },
    {
        "name": "Détecteur de doublons / plagiat (à venir)",
        "description": (
            "Détecter des documents quasi identiques dans une base, ou vérifier si un contenu "
            "a déjà été publié ailleurs. Nécessiterait une recherche de similarité par "
            "embeddings sur l'ensemble d'une base documentaire (au-delà du regroupement par lot "
            "que fait l'entraînement actuel) — pas encore implémenté."
        ),
        "available": False,
    },
]


class ScrollableFrame(ttk.Frame):
    """Conteneur défilable verticalement (molette ou barre de défilement) :
    le contenu réel se construit sur `self.body` plutôt que directement sur
    l'instance. Utilisé par TOUS les onglets, pour qu'aucun contrôle (ex. le
    bouton "Enregistrer" de l'onglet Paramètres, ou "Valider et classer" de
    l'onglet Classification) ne devienne inaccessible sur une fenêtre courte
    ou en plein écran sur un petit moniteur.

    `self.body` reçoit toujours AU MOINS la hauteur visible du canevas (pas
    seulement sa hauteur naturelle) : un contenu qui tiendrait dans la
    fenêtre profite donc toujours pleinement d'un widget interne en
    `fill="both", expand=True` (ex. le tableau de fichiers de l'onglet
    Classification) exactement comme s'il n'y avait pas de défilement du
    tout ; le défilement ne prend le relais que si le contenu dépasse
    réellement l'espace disponible."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.body = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")

        def sync_scrollregion(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_size(event: tk.Event) -> None:
            height = max(self.body.winfo_reqheight(), event.height)
            canvas.itemconfigure(window, width=event.width, height=height)
            sync_scrollregion()

        self.body.bind("<Configure>", sync_scrollregion)
        canvas.bind("<Configure>", sync_size)

        def on_mousewheel(event: tk.Event) -> None:
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        # La molette n'est branchée que pendant que le curseur survole cet
        # onglet, pour ne pas capter le défilement d'un autre onglet actif.
        canvas.bind("<Enter>", lambda _e: (
            canvas.bind_all("<MouseWheel>", on_mousewheel),
            canvas.bind_all("<Button-4>", on_mousewheel),
            canvas.bind_all("<Button-5>", on_mousewheel),
        ))
        canvas.bind("<Leave>", lambda _e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>"),
        ))


class DuplicatesDialog(tk.Toplevel):
    """Affiche les paires de documents quasi identiques détectées, avec un
    bouton pour déplacer les exemplaires en trop vers un dossier `_backup` —
    jamais une suppression définitive. Réutilisé par l'onglet Entraînement
    (après un entraînement où des doublons ont été trouvés) et l'onglet
    Classification (bouton "Détecter les doublons")."""

    def __init__(self, parent: tk.Widget, pairs: list[dict], on_confirm, note: str | None = None):
        super().__init__(parent)
        self.title("Documents en double détectés")
        self.geometry("580x420")
        self.transient(parent.winfo_toplevel())
        self.on_confirm = on_confirm

        ttk.Label(
            self, text=f"{len(pairs)} paire(s) de documents quasi identiques détectée(s) :", padding=(10, 10, 10, 4),
        ).pack(anchor="w")

        list_frame = ttk.Frame(self, padding=(10, 0))
        list_frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(list_frame)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="left", fill="y")
        for pair in pairs:
            listbox.insert("end", f"{pair['filename_a']}  ≈  {pair['filename_b']}   ({pair['similarity']:.0%})")

        if note is None:
            note = (
                "Un exemplaire de chaque groupe de doublons est gardé ; les autres seront déplacés "
                "vers un dossier « _backup » à côté de leur dossier d'origine — jamais supprimés "
                "définitivement, toujours récupérables au besoin."
            )
        ttk.Label(
            self, text=note, foreground="#555", wraplength=540, justify="left", padding=10,
        ).pack(anchor="w")

        actions = ttk.Frame(self, padding=10)
        actions.pack(fill="x")
        ttk.Button(actions, text="Fermer", command=self.destroy).pack(side="right")
        ttk.Button(
            actions, text="Déplacer les doublons vers _backup", command=self._confirm,
        ).pack(side="right", padx=6)

        self.grab_set()

    def _confirm(self) -> None:
        self.destroy()
        self.on_confirm()


_CLUSTER_PREVIEW_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#000000",
]


class ClusterPreviewWindow(tk.Toplevel):
    """Aperçu visuel du regroupement : projette les vecteurs (revectorisés à
    la demande depuis `dataset/`, voir `rename.compute_dataset_vectors`) sur
    un plan 2D via une PCA (analyse en composantes principales — les 2 axes
    qui capturent le plus de variance entre documents), puis dessine un
    point par document, coloré par sa catégorie ACTUELLE (le nom du
    sous-dossier `dataset/` d'où il vient — jamais un identifiant de cluster
    à retraduire, qui pourrait se décaler d'un renommage à l'autre), sur un
    simple `tk.Canvas` (délibérément pas matplotlib, exclu de la variante
    "légère" de l'application — voir build.ps1/build.sh).

    Une projection 2D perd nécessairement de l'information (les distances
    RÉELLES entre documents sont en centaines ou milliers de dimensions) :
    des points proches ici sont probablement proches dans l'espace complet,
    mais l'inverse n'est pas garanti — un outil d'inspection visuelle
    rapide, pas une preuve de la qualité du découpage (voir les métriques
    silhouette/Davies-Bouldin/Calinski-Harabasz affichées dans le journal
    d'entraînement pour ça)."""

    _POINT_RADIUS = 4
    _PADDING = 30

    def __init__(self, parent: tk.Widget, vectors: np.ndarray, labels: list[str]):
        super().__init__(parent)
        self.title("Aperçu du regroupement (projection 2D)")
        self.geometry("820x560")
        self.transient(parent.winfo_toplevel())

        self.vectors = vectors
        self.labels = labels
        self.category_names = sorted(set(labels))
        self._coords: np.ndarray | None = None

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

        legend_container = ScrollableFrame(body, width=220)
        legend_container.pack(side="right", fill="y")
        legend_container.pack_propagate(False)
        self._build_legend(legend_container.body)

        actions = ttk.Frame(self, padding=10)
        actions.pack(fill="x")
        ttk.Label(
            actions,
            text=f"{len(labels)} document(s), {len(self.category_names)} catégorie(s) — projection PCA à 2 axes.",
            foreground="#555",
        ).pack(side="left")
        ttk.Button(actions, text="Fermer", command=self.destroy).pack(side="right")

        self._compute_projection()
        self.grab_set()

    def _color_for(self, category_name: str) -> str:
        index = self.category_names.index(category_name)
        return _CLUSTER_PREVIEW_COLORS[index % len(_CLUSTER_PREVIEW_COLORS)]

    def _build_legend(self, parent: ttk.Frame) -> None:
        for name in self.category_names:
            count = sum(1 for label in self.labels if label == name)
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            swatch = tk.Canvas(row, width=12, height=12, highlightthickness=0)
            swatch.create_oval(1, 1, 11, 11, fill=self._color_for(name), outline=self._color_for(name))
            swatch.pack(side="left", padx=(0, 6))
            ttk.Label(row, text=f"{name} ({count})", wraplength=170, justify="left").pack(side="left", fill="x")

    def _compute_projection(self) -> None:
        if len(self.vectors) < 2 or self.vectors.shape[1] < 2:
            # Un seul document, ou des vecteurs déjà à une seule dimension :
            # rien de significatif à projeter sur 2 axes.
            self._coords = None
            self._redraw()
            return
        from sklearn.decomposition import PCA

        n_components = min(2, self.vectors.shape[0] - 1, self.vectors.shape[1])
        if n_components < 2:
            self._coords = None
            self._redraw()
            return
        self._coords = PCA(n_components=2, random_state=42).fit_transform(self.vectors)
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._coords is None:
            self.canvas.create_text(
                10, 10, anchor="nw",
                text="Pas assez de documents ou de dimensions pour un aperçu 2D.",
            )
            return

        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        pad = self._PADDING
        x_min, y_min = self._coords.min(axis=0)
        x_max, y_max = self._coords.max(axis=0)
        x_range = (x_max - x_min) or 1.0
        y_range = (y_max - y_min) or 1.0

        for (x, y), label in zip(self._coords, self.labels):
            px = pad + (x - x_min) / x_range * (width - 2 * pad)
            # Axe Y inversé : l'origine d'un Canvas Tk est en haut à gauche.
            py = height - pad - (y - y_min) / y_range * (height - 2 * pad)
            color = self._color_for(label)
            r = self._POINT_RADIUS
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")


class EngineHelpDialog(tk.Toplevel):
    """Fenêtre d'aide structurée pour l'onglet Entraînement : pourquoi
    choisir TF-IDF ou embeddings, quel modèle d'embeddings choisir, et ce
    que fait chaque préréglage — regroupé ici plutôt qu'affiché en
    permanence dans le formulaire principal (bouton "? Explications")."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Moteur d'analyse — explications")
        self.geometry("720x600")
        self.transient(parent.winfo_toplevel())

        text = scrolledtext.ScrolledText(self, wrap="word", padx=12, pady=10)
        text.pack(fill="both", expand=True)
        text.tag_configure("h1", font=("TkDefaultFont", 12, "bold"), spacing1=4, spacing3=6)
        text.tag_configure("h2", font=("TkDefaultFont", 10, "bold"), spacing1=10, spacing3=2)
        text.tag_configure("body", font=("TkDefaultFont", 9), spacing3=4)

        text.insert("end", "TF-IDF ou embeddings : lequel choisir ?\n", "h1")
        text.insert("end", ENGINE_EXPLANATION + "\n\n", "body")

        text.insert("end", "Modèles d'embeddings disponibles (léger → lourd)\n", "h1")
        for name, description in EMBEDDING_MODEL_CATALOG:
            text.insert("end", f"{name}\n", "h2")
            text.insert("end", description + "\n", "body")

        text.insert("end", "\nPréréglages disponibles\n", "h1")
        for name, description, _values in TRAINING_PRESETS:
            text.insert("end", f"{name}\n", "h2")
            text.insert("end", description + "\n", "body")

        text.configure(state="disabled")

        ttk.Button(self, text="Fermer", command=self.destroy).pack(pady=(0, 10))
        self.grab_set()


class TrainTab(ttk.Frame):
    """Aucun tri manuel requis : on pointe un dossier de documents en vrac,
    les catégories sont détectées automatiquement (regroupement par
    similarité), comme le mode 'discover' de la CLI."""

    def __init__(self, parent, on_model_created=None):
        super().__init__(parent)
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self.body = scrollable.body
        self.on_model_created = on_model_created
        self.input_dir = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.model_name_var = tk.StringVar()
        self.base_model_path = tk.StringVar()
        self.base_model_picker_var = tk.StringVar()
        self.discovered_base_models: list[tuple[str, int]] = []
        # ── Catégories détectées et fichiers du modèle sélectionné ci-dessus
        # comme base à améliorer (déplacé depuis l'onglet Classification :
        # gérer les catégories AVANT de relancer un entraînement dessus,
        # plutôt que de devoir jongler entre deux onglets) ──
        self.bundle = None
        self.dataset_dir: str | None = None
        self.selected_category_names: list[str] = []
        self.category_status = tk.StringVar(
            value="Sélectionnez un modèle existant à améliorer ci-dessus pour voir ses catégories."
        )
        self.engine = tk.StringVar(value=ENGINE_TFIDF)
        self.embedding_model = tk.StringVar(value=DEFAULT_EMBEDDING_MODEL)
        self.image_model = tk.StringVar(value=DEFAULT_IMAGE_MODEL)
        self.preset_var = tk.StringVar(value=TRAINING_PRESETS[0][0])
        config = get_config()
        self.k_min_var = tk.IntVar(value=config.cluster_k_min)
        self.k_max_var = tk.IntVar(value=config.cluster_k_max)
        self.min_silhouette_var = tk.DoubleVar(value=config.cluster_min_silhouette)
        self.min_cluster_size_var = tk.IntVar(value=config.cluster_min_cluster_size)
        self.tfidf_max_features_var = tk.IntVar(value=config.tfidf_max_features)
        self.tfidf_ngram_max_var = tk.IntVar(value=config.tfidf_ngram_max)
        self.use_keybert_var = tk.BooleanVar(value=config.cluster_use_keybert)
        self.detect_duplicates_var = tk.BooleanVar(value=config.cluster_detect_duplicates)
        self.tfidf_use_stemming_var = tk.BooleanVar(value=config.tfidf_use_stemming)
        self.cluster_algorithm_var = tk.StringVar(value=config.cluster_algorithm)
        self.cluster_use_svd_var = tk.BooleanVar(value=config.cluster_use_svd)
        self.cluster_svd_components_var = tk.IntVar(value=config.cluster_svd_components)
        self.cluster_extra_stopwords_var = tk.StringVar(value=config.cluster_extra_stopwords)
        self.image_cluster_labels_var = tk.StringVar(value=config.image_cluster_labels)
        self.rewrite_config_var = tk.BooleanVar(value=config.train_rewrite_config_default)
        self.use_case_var = tk.StringVar(value=USE_CASES[0]["name"])
        self._last_use_case = USE_CASES[0]["name"]
        self.extension_vars: dict[str, tk.BooleanVar] = {
            ext: tk.BooleanVar(value=True) for _group, exts in EXTENSION_GROUPS for ext in exts
        }
        self.last_model_path: str | None = None
        self._log_row: int = 0
        self._build()
        self.model_name_var.trace_add("write", self._update_resolved_path)
        self.base_model_path.trace_add("write", self._on_base_model_path_changed)
        self._refresh_base_model_picker()

    def _build(self) -> None:
        row = 0
        self._build_use_case_selector(row)
        row += 1

        ttk.Label(self.body, text="1. Dossier de documents à analyser (en vrac, aucun tri préalable requis) :").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1
        ttk.Entry(self.body, textvariable=self.input_dir, width=70).grid(row=row, column=0, columnspan=2, sticky="we")
        ttk.Button(self.body, text="Parcourir...", command=self._choose_input_dir).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Checkbutton(
            self.body, text="Inclure les sous-dossiers", variable=self.recursive_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        row += 1

        self._build_extension_filter(row)
        row += 1

        row = self._build_engine_section(row)
        self._on_engine_change()

        ttk.Label(
            self.body, text="3. Modèle existant à améliorer avec ces nouvelles données (optionnel) :"
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 0))
        row += 1
        base_picker_row = ttk.Frame(self.body)
        base_picker_row.grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Label(base_picker_row, text="Modèles disponibles (du plus léger au plus lourd) :").pack(side="left")
        self.base_model_picker = ttk.Combobox(
            base_picker_row, textvariable=self.base_model_picker_var, state="readonly", width=50
        )
        self.base_model_picker.pack(side="left", padx=6)
        self.base_model_picker.bind("<<ComboboxSelected>>", self._on_pick_base_model)
        ttk.Button(base_picker_row, text="Rafraîchir", command=self._refresh_base_model_picker).pack(side="left")
        row += 1
        ttk.Entry(self.body, textvariable=self.base_model_path, width=70).grid(row=row, column=0, columnspan=2, sticky="we")
        ttk.Button(self.body, text="Parcourir...", command=self._choose_base_model).grid(row=row, column=2, padx=4)
        row += 1
        ttk.Button(self.body, text="Aucun (nouveau modèle)", command=lambda: self.base_model_path.set("")).grid(
            row=row, column=0, sticky="w", pady=(2, 0)
        )
        ttk.Checkbutton(
            self.body, text="Réécrire la configuration (reprendre les réglages de ce modèle)",
            variable=self.rewrite_config_var, command=self._on_rewrite_config_toggle,
        ).grid(row=row, column=1, columnspan=2, sticky="w", pady=(2, 0))
        row += 1
        self.history_button = ttk.Button(
            self.body, text="Historique / Revenir en arrière...", command=self._open_history_dialog,
            state="disabled",
        )
        self.history_button.grid(row=row, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            self.body,
            text="Chaque entraînement/amélioration archive l'état précédent du modèle ci-dessus.",
            foreground="#555",
        ).grid(row=row, column=1, columnspan=2, sticky="w", pady=(2, 0))
        row += 1
        self.delete_model_button = ttk.Button(
            self.body, text="Supprimer ce modèle (définitif)", command=self._delete_model_permanently,
            state="disabled",
        )
        self.delete_model_button.grid(row=row, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            self.body,
            text="Supprime le .pkl, son .json, dataset/ et son historique — jamais les documents source.",
            foreground="#555",
        ).grid(row=row, column=1, columnspan=2, sticky="w", pady=(2, 0))
        row += 1

        row = self._build_categories_section(row)

        ttk.Label(self.body, text="4. Nom du modèle :").grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 0))
        row += 1
        ttk.Entry(self.body, textvariable=self.model_name_var, width=40).grid(row=row, column=0, sticky="w")
        self.resolved_path_var = tk.StringVar()
        ttk.Label(self.body, textvariable=self.resolved_path_var, foreground="#555").grid(
            row=row, column=1, columnspan=2, sticky="w", padx=(8, 0)
        )
        row += 1

        action_row = ttk.Frame(self.body)
        action_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.train_button = ttk.Button(action_row, text="Entraîner le modèle", command=self._start_training)
        self.train_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            action_row, text="Ouvrir le dossier du modèle", command=self._open_preview_dir, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=6)
        self.cluster_preview_button = ttk.Button(
            action_row, text="Aperçu du regroupement (2D)", command=self._show_cluster_preview, state="disabled"
        )
        self.cluster_preview_button.pack(side="left", padx=6)
        row += 1

        self.progress_bar = ttk.Progressbar(self.body, mode="indeterminate")
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky="we", pady=(0, 8))
        row += 1

        self.log = scrolledtext.ScrolledText(self.body, height=14, state="disabled")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self._log_row = row

        self.body.grid_rowconfigure(row, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

    def _build_use_case_selector(self, row: int) -> None:
        frame = ttk.LabelFrame(self.body, text="Cas d'usage (facultatif)", padding=10)
        frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(0, 10))

        ttk.Label(frame, text="Pré-remplir le formulaire pour :").grid(row=0, column=0, sticky="w")
        use_case_combo = ttk.Combobox(
            frame, textvariable=self.use_case_var,
            values=[uc["name"] for uc in USE_CASES],
            state="readonly", width=55,
        )
        use_case_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        use_case_combo.bind("<<ComboboxSelected>>", self._apply_use_case)

        self.use_case_description_var = tk.StringVar(value=USE_CASES[0]["description"])
        ttk.Label(
            frame, textvariable=self.use_case_description_var, foreground="#555",
            wraplength=720, justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _apply_use_case(self, _event=None) -> None:
        name = self.use_case_var.get()
        use_case = next((uc for uc in USE_CASES if uc["name"] == name), None)
        if use_case is None:
            return
        self.use_case_description_var.set(use_case["description"])

        if not use_case["available"]:
            messagebox.showinfo(name, use_case["description"])
            self.use_case_var.set(self._last_use_case)
            previous = next(uc for uc in USE_CASES if uc["name"] == self._last_use_case)
            self.use_case_description_var.set(previous["description"])
            return

        self._last_use_case = name
        extensions = use_case["extensions"]
        for ext, var in self.extension_vars.items():
            var.set(extensions is None or ext in extensions)

        self.preset_var.set(use_case["preset"])
        self._apply_preset()

    def _build_extension_filter(self, row: int) -> None:
        frame = ttk.LabelFrame(self.body, text="Types de fichiers à inclure", padding=10)
        frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Button(buttons, text="Tout sélectionner", command=lambda: self._set_all_extensions(True)).pack(
            side="left"
        )
        ttk.Button(buttons, text="Aucun", command=lambda: self._set_all_extensions(False)).pack(
            side="left", padx=(6, 0)
        )

        for group_row, (group_name, extensions) in enumerate(EXTENSION_GROUPS, start=1):
            ttk.Label(frame, text=f"{group_name} :").grid(row=group_row, column=0, sticky="nw", pady=2)
            checks = ttk.Frame(frame)
            checks.grid(row=group_row, column=1, sticky="w", pady=2)
            for ext in extensions:
                ttk.Checkbutton(checks, text=ext, variable=self.extension_vars[ext]).pack(side="left", padx=(0, 8))

    def _set_all_extensions(self, value: bool) -> None:
        for var in self.extension_vars.values():
            var.set(value)

    def _selected_extensions(self) -> tuple[str, ...]:
        return tuple(ext for ext, var in self.extension_vars.items() if var.get())

    def _on_engine_change(self, _event=None) -> None:
        engine = self.engine.get()
        is_tfidf = engine == ENGINE_TFIDF
        is_embeddings = engine == ENGINE_EMBEDDINGS
        is_image = engine == ENGINE_IMAGE

        self.embedding_combo.configure(state="readonly" if is_embeddings else "disabled")
        self.image_model_combo.configure(state="readonly" if is_image else "disabled")
        # Jamais affichés en même temps : le modèle d'embeddings TEXTE et le
        # modèle CLIP (analyse visuelle) partagent la même ligne du
        # formulaire, seul l'un des deux a un sens selon le moteur choisi.
        if is_image:
            self.embedding_combo.grid_remove()
            self.image_model_combo.grid()
        else:
            self.image_model_combo.grid_remove()
            self.embedding_combo.grid()

        # Racinisation, n-grammes et réduction SVD n'ont d'effet qu'en
        # TF-IDF (voir discover._build_bundle) : grisés avec le reste des
        # réglages TF-IDF plutôt que laissés actifs sans effet visible avec
        # "embeddings"/"image".
        tfidf_state = "normal" if is_tfidf else "disabled"
        self.tfidf_max_features_spin.configure(state=tfidf_state)
        self.tfidf_ngram_max_spin.configure(state=tfidf_state)
        self.tfidf_stemming_check.configure(state=tfidf_state)
        self.cluster_svd_check.configure(state=tfidf_state)
        self.cluster_svd_components_spin.configure(state=tfidf_state)

        self.image_labels_entry.configure(state="normal" if is_image else "disabled")

    def _build_engine_section(self, row: int) -> int:
        """"2. Moteur d'analyse" : le préréglage choisi juste en dessous du
        titre détermine le moteur ET tous ses paramètres, montrés ensuite —
        un préréglage peut toujours être affiné en modifiant un champ
        individuellement après coup. Les explications complètes (pourquoi
        TF-IDF vs embeddings, ce que fait chaque préréglage...) sont dans une
        fenêtre dédiée plutôt qu'affichées en permanence ici (bouton
        "Explications")."""
        header = ttk.Frame(self.body)
        header.grid(row=row, column=0, columnspan=3, sticky="we", pady=(14, 0))
        ttk.Label(header, text="2. Moteur d'analyse :").pack(side="left")
        ttk.Button(header, text="? Explications", command=self._show_engine_help).pack(side="right")
        row += 1

        frame = ttk.Frame(self.body)
        frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(6, 0))
        row += 1

        ttk.Label(frame, text="Préréglage :").grid(row=0, column=0, sticky="w")
        preset_combo = ttk.Combobox(
            frame, textvariable=self.preset_var,
            values=[name for name, _description, _values in TRAINING_PRESETS],
            state="readonly", width=62,
        )
        preset_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=(6, 0))
        preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)

        ttk.Label(frame, text="Moteur :").grid(row=1, column=0, sticky="w", pady=(10, 0))
        engine_combo = ttk.Combobox(
            frame, textvariable=self.engine,
            values=[ENGINE_TFIDF, ENGINE_EMBEDDINGS, ENGINE_IMAGE], state="readonly", width=15,
        )
        engine_combo.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(10, 0))
        engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

        self._model_combo_row, self._model_combo_column = 2, 1
        ttk.Label(frame, text="Modèle d'embeddings :").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.embedding_combo = ttk.Combobox(
            frame, textvariable=self.embedding_model,
            values=[name for name, _description in EMBEDDING_MODEL_CATALOG],
            state="readonly", width=45,
        )
        self.embedding_combo.grid(
            row=self._model_combo_row, column=self._model_combo_column, columnspan=3, sticky="w",
            padx=(6, 0), pady=(4, 0),
        )
        # Même ligne que "Modèle d'embeddings" ci-dessus (jamais affichés en
        # même temps, voir `_on_engine_change`) : un modèle CLIP (analyse
        # VISUELLE — personnes, scènes, objets) pour le moteur "image", pas
        # le texte imprimé sur l'image (ça, c'est l'OCR — voir Paramètres).
        self.image_model_combo = ttk.Combobox(
            frame, textvariable=self.image_model,
            values=[name for name, _description in IMAGE_MODEL_CATALOG],
            state="readonly", width=45,
        )
        self.image_model_combo.grid(
            row=self._model_combo_row, column=self._model_combo_column, columnspan=3, sticky="w",
            padx=(6, 0), pady=(4, 0),
        )
        self.image_model_combo.grid_remove()

        def spin(spin_row, col, label, var, from_, to, increment=1):
            ttk.Label(frame, text=label).grid(
                row=spin_row, column=col * 2, sticky="w", padx=(0 if col == 0 else 16, 0), pady=(10, 0)
            )
            box = ttk.Spinbox(frame, from_=from_, to=to, increment=increment, textvariable=var, width=8)
            box.grid(row=spin_row, column=col * 2 + 1, sticky="w", padx=(4, 0), pady=(10, 0))
            return box

        spin(3, 0, "Nb minimal de catégories :", self.k_min_var, 1, 100)
        spin(3, 1, "Nb maximal de catégories :", self.k_max_var, 1, 200)
        spin(4, 0, "Score de silhouette minimal :", self.min_silhouette_var, 0.0, 1.0, 0.01)
        spin(4, 1, "Documents minimum par catégorie :", self.min_cluster_size_var, 1, 50)
        self.tfidf_max_features_spin = spin(5, 0, "Vocabulaire TF-IDF maximal :", self.tfidf_max_features_var, 100, 50000, 100)
        self.tfidf_ngram_max_spin = spin(5, 1, "Taille max des n-grammes :", self.tfidf_ngram_max_var, 1, 4)

        ttk.Checkbutton(
            frame, text="Noms de catégorie plus naturels (KeyBERT)", variable=self.use_keybert_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            frame, text="Détecter les documents en double", variable=self.detect_duplicates_var,
        ).grid(row=6, column=2, columnspan=2, sticky="w", pady=(10, 0))
        self.tfidf_stemming_check = ttk.Checkbutton(
            frame, text="Racinisation des mots (FR/EN)", variable=self.tfidf_use_stemming_var,
        )
        self.tfidf_stemming_check.grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.cluster_svd_check = ttk.Checkbutton(
            frame, text="Réduction de dimension avant regroupement (SVD)", variable=self.cluster_use_svd_var,
        )
        self.cluster_svd_check.grid(row=7, column=2, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(frame, text="Algorithme de regroupement :").grid(row=8, column=0, sticky="w", pady=(10, 0))
        self.cluster_algorithm_combo = ttk.Combobox(
            frame, textvariable=self.cluster_algorithm_var,
            values=["kmeans", "minibatch_kmeans", "agglomerative", "hdbscan"], state="readonly", width=15,
        )
        self.cluster_algorithm_combo.grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(10, 0))
        self.cluster_svd_components_spin = spin(8, 1, "Composantes SVD :", self.cluster_svd_components_var, 2, 2000, 10)

        ttk.Label(frame, text="Mots à ignorer en plus (virgules) :").grid(row=9, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.cluster_extra_stopwords_var, width=48).grid(
            row=9, column=1, columnspan=3, sticky="we", padx=(6, 0), pady=(10, 0)
        )

        self.image_labels_label = ttk.Label(frame, text="Libellés candidats (moteur image, virgules) :")
        self.image_labels_label.grid(row=10, column=0, sticky="w", pady=(10, 0))
        self.image_labels_entry = ttk.Entry(frame, textvariable=self.image_cluster_labels_var, width=48)
        self.image_labels_entry.grid(row=10, column=1, columnspan=3, sticky="we", padx=(6, 0), pady=(10, 0))

        return row

    def _show_engine_help(self) -> None:
        EngineHelpDialog(self)

    def _apply_preset(self, _event=None) -> None:
        name = self.preset_var.get()
        match = next((p for p in TRAINING_PRESETS if p[0] == name), None)
        if match is None:
            return
        values = match[2]

        config = get_config()
        self.engine.set(values.get("engine", ENGINE_TFIDF))
        self.embedding_model.set(values.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        self.image_model.set(values.get("image_model", DEFAULT_IMAGE_MODEL))
        self.image_cluster_labels_var.set(values.get("image_cluster_labels", config.image_cluster_labels))
        self.k_min_var.set(values.get("k_min", config.cluster_k_min))
        self.k_max_var.set(values.get("k_max", config.cluster_k_max))
        self.min_silhouette_var.set(values.get("cluster_min_silhouette", config.cluster_min_silhouette))
        self.min_cluster_size_var.set(values.get("cluster_min_cluster_size", config.cluster_min_cluster_size))
        self.tfidf_max_features_var.set(values.get("tfidf_max_features", config.tfidf_max_features))
        self.tfidf_ngram_max_var.set(values.get("tfidf_ngram_max", config.tfidf_ngram_max))
        self.tfidf_use_stemming_var.set(values.get("tfidf_use_stemming", config.tfidf_use_stemming))
        self.cluster_algorithm_var.set(values.get("cluster_algorithm", config.cluster_algorithm))
        self.cluster_use_svd_var.set(values.get("cluster_use_svd", config.cluster_use_svd))
        self.cluster_svd_components_var.set(values.get("cluster_svd_components", config.cluster_svd_components))
        self.cluster_extra_stopwords_var.set(values.get("cluster_extra_stopwords", config.cluster_extra_stopwords))
        self.use_keybert_var.set(values.get("use_keybert", config.cluster_use_keybert))
        self.detect_duplicates_var.set(values.get("detect_duplicates", config.cluster_detect_duplicates))

        self._on_engine_change()

    def _choose_input_dir(self) -> None:
        kwargs = {"title": "Dossier de documents"}
        initial = self.input_dir.get().strip()
        if not initial and not self.base_model_path.get().strip():
            # Aucun modèle existant sélectionné pour cet entraînement :
            # ouvrir directement sur le répertoire d'installation de
            # l'application (où vivent storage/models/, config.json...)
            # plutôt que de laisser l'explorateur choisir un dossier sans
            # rapport avec l'outil — utile notamment pour chaîner un modèle
            # depuis le dataset/ d'un autre (voir l'étape 1 du README).
            initial = os.getcwd()
        if initial:
            kwargs["initialdir"] = initial
        directory = filedialog.askdirectory(**kwargs)
        if directory:
            self.input_dir.set(directory)

    def _refresh_base_model_picker(self) -> None:
        """Propose les `.pkl` déjà trouvés dans le projet (même recherche que
        la liste "Modèles disponibles" de l'onglet Classification), pour ne
        pas devoir systématiquement parcourir le disque à la main pour un
        modèle déjà connu de l'outil."""
        self.discovered_base_models = model_store.discover_models(".")
        display_values = [
            f"{os.path.relpath(path)}  ({_human_size(size)})" for path, size in self.discovered_base_models
        ]
        self.base_model_picker.configure(values=display_values)
        if not display_values:
            self.base_model_picker_var.set("(aucun modèle .pkl trouvé dans le projet)")

    def _on_pick_base_model(self, _event=None) -> None:
        index = self.base_model_picker.current()
        if index < 0 or index >= len(self.discovered_base_models):
            return
        path, _size = self.discovered_base_models[index]
        self._select_base_model(path)

    def _choose_base_model(self) -> None:
        path = filedialog.askopenfilename(title="Modèle existant à améliorer", filetypes=[("Modèle entraîné", "*.pkl")])
        if not path:
            return
        self._select_base_model(path)

    def _select_base_model(self, path: str) -> None:
        """Applique le choix d'un modèle de base — que ce soit via le menu
        déroulant ou "Parcourir..." — de la même façon dans les deux cas."""
        self.base_model_path.set(path)
        if not self.model_name_var.get().strip():
            # Par défaut, on améliore ce modèle en place : même nom.
            self.model_name_var.set(os.path.splitext(os.path.basename(path))[0])
        if self.rewrite_config_var.get():
            self._apply_model_config(path)

    def _on_rewrite_config_toggle(self) -> None:
        """Coché alors qu'un modèle de base est déjà sélectionné : recharge
        tout de suite sa configuration, sans attendre un nouveau choix de
        fichier via "Parcourir..."."""
        path = self.base_model_path.get().strip()
        if self.rewrite_config_var.get() and path:
            self._apply_model_config(path)

    def _apply_model_config(self, model_path: str) -> None:
        """Recharge dans le formulaire les réglages EXACTS (moteur, k_min/
        k_max, types de fichiers, sous-dossiers...) que `model_path` avait
        lors de son dernier entraînement (`training_params`/`engine_state`,
        voir `discover._build_bundle`), plutôt que de laisser le formulaire
        sur des valeurs potentiellement différentes — un préréglage choisi
        entre-temps, par exemple — qui écraseraient silencieusement la
        configuration d'origine du modèle lors de son amélioration."""
        try:
            bundle = model_store.load_bundle(model_path)
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de lire la configuration de ce modèle :\n{exc}")
            return

        params = bundle.get("training_params", {})
        engine_state = bundle.get("engine_state", {})
        config = get_config()

        engine_type = engine_state.get("type", ENGINE_TFIDF)
        self.engine.set(engine_type)
        if engine_type == ENGINE_IMAGE:
            self.image_model.set(engine_state.get("model_name") or DEFAULT_IMAGE_MODEL)
        else:
            self.embedding_model.set(engine_state.get("model_name") or DEFAULT_EMBEDDING_MODEL)
        self.image_cluster_labels_var.set(params.get("image_cluster_labels", config.image_cluster_labels))
        self.k_min_var.set(params.get("k_min", config.cluster_k_min))
        self.k_max_var.set(params.get("k_max", config.cluster_k_max))
        self.min_silhouette_var.set(params.get("min_silhouette", config.cluster_min_silhouette))
        self.min_cluster_size_var.set(params.get("min_cluster_size", config.cluster_min_cluster_size))
        self.tfidf_max_features_var.set(params.get("tfidf_max_features", config.tfidf_max_features))
        self.tfidf_ngram_max_var.set(params.get("tfidf_ngram_max", config.tfidf_ngram_max))
        self.use_keybert_var.set(params.get("use_keybert", config.cluster_use_keybert))
        self.tfidf_use_stemming_var.set(params.get("tfidf_use_stemming", config.tfidf_use_stemming))
        self.cluster_algorithm_var.set(params.get("cluster_algorithm", config.cluster_algorithm))
        self.cluster_use_svd_var.set(params.get("cluster_use_svd", config.cluster_use_svd))
        self.cluster_svd_components_var.set(params.get("cluster_svd_components", config.cluster_svd_components))
        self.cluster_extra_stopwords_var.set(params.get("cluster_extra_stopwords", config.cluster_extra_stopwords))
        self.recursive_var.set(params.get("recursive", True))

        extensions = params.get("extensions")
        if extensions:
            ext_set = set(extensions)
            for ext, var in self.extension_vars.items():
                var.set(ext in ext_set)
        else:
            self._set_all_extensions(True)

        self._on_engine_change()
        # Aucun des 10 préréglages ne correspond forcément exactement à cette
        # combinaison reprise du modèle : l'afficher tel quel plutôt que de
        # laisser un nom de préréglage trompeur (qui ne reflèterait plus les
        # champs réellement affichés).
        self.preset_var.set("(configuration reprise du modèle sélectionné)")
        self._log_line(f"✓ Configuration reprise du modèle : {model_path}")

    # ── Catégories et fichiers du modèle choisi ci-dessus comme base à
    # améliorer (déplacé depuis l'onglet Classification) : renommer,
    # déplacer des fichiers, ou supprimer une catégorie. Seuls les modèles
    # non supervisés (catégories détectées automatiquement) le permettent ──
    def _on_base_model_path_changed(self, *_args) -> None:
        """Recharge les catégories du modèle choisi comme base à améliorer, à
        chaque changement de `base_model_path` (saisie manuelle,
        "Parcourir...", ou "Aucun (nouveau modèle)"). Volontairement
        silencieux en cas de chemin invalide ou incomplet — ce callback se
        déclenche à chaque frappe clavier dans le champ, pas seulement lors
        d'un choix explicite via le sélecteur de fichier."""
        path = self.base_model_path.get().strip()
        if not path or not os.path.exists(path):
            self.bundle = None
            self.dataset_dir = None
            self._populate_categories()
            return
        try:
            self.bundle = model_store.load_bundle(path)
        except Exception:
            self.bundle = None
            self.dataset_dir = None
            self._populate_categories()
            return
        self.dataset_dir = model_store.model_dataset_dir(path)
        self._populate_categories()

    def _build_categories_section(self, row: int) -> int:
        """Catégories détectées et fichiers du modèle choisi comme base à
        améliorer ci-dessus : renommer, déplacer des fichiers d'une
        catégorie à une autre, ou supprimer une catégorie. Seuls les
        modèles "non supervisés" (catégories détectées automatiquement) le
        permettent — la section reste vide sinon. Se rafraîchit
        automatiquement à chaque changement de `base_model_path` (voir
        `_on_base_model_path_changed`)."""
        frame = ttk.Frame(self.body)
        frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        row += 1

        ttk.Label(
            frame, text="Catégories de ce modèle (sélection multiple possible, Ctrl/Shift+clic) :"
        ).pack(anchor="w", pady=(0, 6))

        categories_paned = ttk.PanedWindow(frame, orient="horizontal", height=220)
        categories_paned.pack(fill="x")

        tree_frame = ttk.Frame(categories_paned)
        categories_paned.add(tree_frame, weight=2)
        self.category_tree = ttk.Treeview(
            tree_frame, columns=("name", "detected", "count"), show="headings", selectmode="extended"
        )
        self.category_tree.heading("name", text="Catégorie")
        self.category_tree.heading("detected", text="Nom détecté par le modèle")
        self.category_tree.heading("count", text="Fichiers")
        self.category_tree.column("name", width=200)
        self.category_tree.column("detected", width=200)
        self.category_tree.column("count", width=70, anchor="center")
        self.category_tree.pack(side="left", fill="both", expand=True)
        self.category_tree.bind("<<TreeviewSelect>>", self._on_select_category)

        category_tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.category_tree.yview)
        self.category_tree.configure(yscrollcommand=category_tree_scrollbar.set)
        category_tree_scrollbar.pack(side="left", fill="y")

        details = ttk.Frame(categories_paned, padding=(10, 0, 0, 0))
        categories_paned.add(details, weight=3)

        rename_row = ttk.Frame(details)
        rename_row.pack(fill="x")
        ttk.Label(rename_row, text="Nouveau nom (plusieurs catégories sélectionnées = fusion) :").pack(side="left")
        self.rename_var = tk.StringVar()
        self.rename_entry = ttk.Entry(rename_row, textvariable=self.rename_var, width=32, state="disabled")
        self.rename_entry.pack(side="left", padx=6)
        self.rename_button = ttk.Button(rename_row, text="Renommer", command=self._rename_selected, state="disabled")
        self.rename_button.pack(side="left")

        ttk.Label(details, text="Fichiers de cette catégorie (sélection multiple possible) :").pack(
            anchor="w", pady=(8, 2)
        )
        files_frame = ttk.Frame(details)
        files_frame.pack(fill="both", expand=True)
        self.category_files_listbox = tk.Listbox(files_frame, selectmode="extended", height=6)
        self.category_files_listbox.pack(side="left", fill="both", expand=True)
        category_files_scrollbar = ttk.Scrollbar(
            files_frame, orient="vertical", command=self.category_files_listbox.yview
        )
        self.category_files_listbox.configure(yscrollcommand=category_files_scrollbar.set)
        category_files_scrollbar.pack(side="left", fill="y")

        move_row = ttk.Frame(details)
        move_row.pack(fill="x", pady=(4, 0))
        ttk.Label(move_row, text="Déplacer la sélection vers :").pack(side="left")
        self.move_target_var = tk.StringVar()
        self.move_target_combo = ttk.Combobox(move_row, textvariable=self.move_target_var, width=22)
        self.move_target_combo.pack(side="left", padx=6)
        self.move_files_button = ttk.Button(
            move_row, text="Déplacer", command=self._move_selected_files, state="disabled"
        )
        self.move_files_button.pack(side="left")

        # Sens inverse de "Déplacer" : au lieu de partir d'un fichier déjà
        # classé pour lui choisir sa catégorie, partir de la catégorie
        # sélectionnée (une seule) pour lui affecter directement un ou
        # plusieurs fichiers pris n'importe où sur le disque.
        self.add_files_to_category_button = ttk.Button(
            move_row, text="Ajouter des fichiers à cette catégorie...", command=self._add_files_to_selected_category,
            state="disabled",
        )
        self.add_files_to_category_button.pack(side="left", padx=(18, 0))
        self.delete_files_button = ttk.Button(
            move_row, text="Supprimer définitivement la sélection", command=self._delete_selected_files,
            state="disabled",
        )
        self.delete_files_button.pack(side="left", padx=(18, 0))

        category_actions = ttk.Frame(details)
        category_actions.pack(fill="x", pady=(6, 0))
        self.open_category_folder_button = ttk.Button(
            category_actions, text="Ouvrir le dossier", command=self._open_category_folder, state="disabled"
        )
        self.open_category_folder_button.pack(side="left")
        self.prefix_button = ttk.Button(
            category_actions, text="Préfixer les fichiers par la catégorie", command=self._prefix_files,
            state="disabled",
        )
        self.prefix_button.pack(side="left", padx=6)
        self.delete_category_button = ttk.Button(
            category_actions, text="Supprimer la sélection (→ autre)", command=self._delete_category_selected,
            state="disabled",
        )
        self.delete_category_button.pack(side="left")
        self.delete_category_permanently_button = ttk.Button(
            category_actions, text="Supprimer définitivement (irréversible)",
            command=self._delete_category_permanently_selected, state="disabled",
        )
        self.delete_category_permanently_button.pack(side="left", padx=6)

        ttk.Label(frame, textvariable=self.category_status, foreground="#555").pack(anchor="w", pady=(4, 0))

        return row

    def _open_history_dialog(self) -> None:
        model_path = self.base_model_path.get().strip()
        if not model_path or not os.path.exists(model_path):
            return
        HistoryDialog(self, model_path, on_restored=self._on_history_restored)

    def _on_history_restored(self, model_path: str) -> None:
        # Recharge tout ce qui dépend de ce modèle (catégories, aperçu,
        # historique lui-même) depuis son état restauré, et propage jusqu'à
        # l'onglet Classification (voir `App._on_model_trained`) — une
        # restauration change le contenu du modèle exactement comme un
        # entraînement, elle doit donc être suivie des mêmes rafraîchissements.
        self._on_base_model_path_changed()
        if self.on_model_created:
            self.on_model_created(model_path)

    def _delete_model_permanently(self) -> None:
        """Supprime DÉFINITIVEMENT le modèle actuellement chargé ci-dessus
        (voir `model_store.delete_model_permanently`) : le .pkl, son .json,
        son dataset/ et tout son historique — AUCUN instantané n'est pris
        avant (contrairement aux suppressions de catégorie/fichier
        ci-dessous), il n'y aurait nulle part où le restaurer une fois ces
        fichiers eux-mêmes supprimés. Les documents SOURCE, eux, ne sont
        jamais touchés."""
        model_path = self.base_model_path.get().strip()
        if not model_path or not os.path.exists(model_path):
            return
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        if not messagebox.askyesno(
            "Confirmer la suppression définitive du modèle",
            f"Supprimer définitivement le modèle « {model_name} » ?\n\n{model_path}\n\n"
            "Le .pkl, son .json, son dossier dataset/ ET tout son historique (aucun retour en "
            "arrière possible ensuite) seront supprimés. Les documents d'origine, eux, ne "
            "sont jamais touchés.",
        ):
            return
        try:
            model_store.delete_model_permanently(model_path)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return
        self.base_model_path.set("")
        self._refresh_base_model_picker()
        if self.last_model_path and os.path.abspath(self.last_model_path) == os.path.abspath(model_path):
            self.last_model_path = None
            self.open_folder_button.configure(state="disabled")
            self.cluster_preview_button.configure(state="disabled")
        messagebox.showinfo("Terminé", f"Modèle « {model_name} » supprimé définitivement.")

    def _populate_categories(self) -> None:
        self.category_tree.delete(*self.category_tree.get_children())
        self.history_button.configure(state="normal" if self.bundle else "disabled")
        self.delete_model_button.configure(state="normal" if self.bundle else "disabled")
        if not self.bundle or self.bundle.get("mode") != "unsupervised":
            self._clear_category_selection_ui()
            self.cluster_preview_button.configure(state="disabled")
            self.category_status.set(
                "Catégories modifiables uniquement pour les modèles à catégories détectées "
                "automatiquement." if self.bundle else
                "Sélectionnez un modèle existant à améliorer ci-dessus pour voir ses catégories."
            )
            return

        cluster_names = self.bundle["cluster_names"]
        original_names = self.bundle.get("original_cluster_names", {})
        has_dataset = bool(self.dataset_dir and os.path.isdir(self.dataset_dir))

        # Pour une catégorie confirmée à la main sans cluster K-Means (voir
        # plus bas), le nom "détecté" n'a pas de cluster_id à consulter dans
        # original_cluster_names — mais chaque fichier du manifest garde son
        # propre `detected_category` (voir `discover._sync_dataset`), donc on
        # peut quand même afficher ce que le moteur avait trouvé pour LUI,
        # plutôt que la mention générique "(confirmée manuellement)".
        manifest_detected_by_category: dict[str, set[str]] = {}
        manifest = model_store.load_manifest(self.base_model_path.get())
        for entry in manifest.get("files", {}).values():
            if not isinstance(entry, dict):
                continue
            detected, category = entry.get("detected_category"), entry.get("category")
            if detected and category:
                manifest_detected_by_category.setdefault(category, set()).add(detected)

        # Plusieurs identifiants de cluster internes peuvent partager le même
        # nom affiché (ex. après plusieurs améliorations successives, ou un
        # renommage manuel qui rapproche deux clusters) : regrouper par nom
        # pour que chaque catégorie n'apparaisse qu'une seule fois. Les noms
        # "détectés" d'origine, potentiellement différents entre clusters
        # fusionnés, sont listés ensemble pour rester transparent.
        by_name: dict[str, list[int]] = {}
        for cluster_id, name in cluster_names.items():
            by_name.setdefault(name, []).append(cluster_id)
        # Une catégorie confirmée à la main (correction dans le tableau
        # ci-dessous) peut ne correspondre à AUCUN cluster K-Means (le
        # clustering reste non supervisé) : sans cette ligne, ses documents
        # existeraient bien dans dataset/ mais la catégorie n'apparaîtrait
        # jamais ici.
        for name in self.bundle.get("confirmed_overrides", {}).values():
            by_name.setdefault(name, [])

        for name in sorted(by_name):
            cluster_ids = by_name[name]
            detected = sorted({original_names.get(cid, "n/a") for cid in cluster_ids})
            if not detected:
                detected = sorted(manifest_detected_by_category.get(name, set()))
            detected_display = " / ".join(detected) if detected else "(confirmée manuellement)"
            count = len(list_category_files(self.dataset_dir, name)) if has_dataset else None
            self.category_tree.insert(
                "", "end", iid=name,
                values=(name, detected_display, count if count is not None else "n/a"),
            )
        self.move_target_combo.configure(values=sorted(by_name))
        # L'aperçu (voir `_show_cluster_preview`) revectorise à la demande les
        # fichiers de dataset/ : possible dès qu'un dossier dataset existe,
        # que ce modèle vienne d'être (ré)entraîné ou simplement rechargé.
        self.cluster_preview_button.configure(state="normal" if has_dataset else "disabled")

        dataset_note = (
            " (dossier dataset trouvé — fichiers consultables)"
            if has_dataset
            else " (dossier dataset pas encore créé — entraînez ou améliorez ce modèle une première fois)"
        )
        self.category_status.set(f"{len(by_name)} catégorie(s){dataset_note}.")
        self._clear_category_selection_ui()

    def _clear_category_selection_ui(self) -> None:
        self.selected_category_names = []
        self.rename_var.set("")
        self.rename_entry.configure(state="disabled")
        self.rename_button.configure(state="disabled")
        self.category_files_listbox.delete(0, "end")
        self.open_category_folder_button.configure(state="disabled")
        self.prefix_button.configure(state="disabled")
        self.delete_category_button.configure(state="disabled")
        self.delete_category_permanently_button.configure(state="disabled")
        self.move_files_button.configure(state="disabled")
        self.add_files_to_category_button.configure(state="disabled")
        self.delete_files_button.configure(state="disabled")

    def _on_select_category(self, _event=None) -> None:
        """Sélection simple ou multiple (Ctrl/Shift+clic, comme le tableau
        des documents à classer). Renommer/supprimer s'appliquent à TOUTES
        les catégories sélectionnées à la fois (renommer plusieurs vers le
        même nouveau nom les fusionne) ; la liste de fichiers et le
        déplacement de fichiers précis, eux, n'ont de sens que pour UNE
        seule catégorie source à la fois."""
        selection = list(self.category_tree.selection())
        if not selection or not self.bundle:
            self._clear_category_selection_ui()
            return

        self.selected_category_names = selection
        has_dataset = bool(self.dataset_dir and os.path.isdir(self.dataset_dir))
        single = len(selection) == 1

        self.category_files_listbox.delete(0, "end")
        if single and has_dataset:
            for filename in list_category_files(self.dataset_dir, selection[0]):
                self.category_files_listbox.insert("end", filename)
        self.move_files_button.configure(state="normal" if (single and has_dataset) else "disabled")
        self.delete_files_button.configure(state="normal" if (single and has_dataset) else "disabled")
        # Contrairement à "Déplacer" (qui a besoin d'un dataset/ déjà créé
        # pour cette catégorie), ajouter des fichiers externes fonctionne
        # même pour une catégorie toute neuve sans dossier existant.
        self.add_files_to_category_button.configure(state="normal" if single else "disabled")

        # Pré-remplir avec le nom sélectionné n'a de sens que pour une seule
        # catégorie ; pour plusieurs, un champ vide évite de laisser croire
        # que le nom de la première s'appliquerait telle quelle à la fusion.
        self.rename_var.set(selection[0] if single else "")
        self.rename_entry.configure(state="normal")
        self.rename_button.configure(state="normal")
        self.open_category_folder_button.configure(state="normal" if has_dataset else "disabled")
        self.prefix_button.configure(state="normal" if has_dataset else "disabled")

        other_name = get_config().other_category_name
        all_names = set(self.bundle["cluster_names"].values()) | set(self.bundle.get("confirmed_overrides", {}).values())
        can_delete = len(all_names) > 1 and any(name != other_name for name in selection)
        self.delete_category_button.configure(state="normal" if can_delete else "disabled")
        # Contrairement à "Supprimer la sélection (→ autre)" ci-dessus (qui
        # fusionne, jamais de perte), la suppression définitive n'a pas
        # besoin d'exclure "autre" ni d'exiger une autre catégorie restante
        # — l'utilisateur choisit explicitement de perdre ces documents.
        self.delete_category_permanently_button.configure(state="normal" if has_dataset else "disabled")

    def _rename_selected(self) -> None:
        names = self.selected_category_names
        if not names:
            return
        new_name = self.rename_var.get().strip()
        if not new_name:
            messagebox.showwarning("Nom vide", "Le nom ne peut pas être vide.")
            return
        renames = {name: new_name for name in names if name != new_name}
        if not renames:
            return
        try:
            bundle = rename_categories(self.base_model_path.get(), renames)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = bundle
        self._populate_categories()
        if len(renames) == 1:
            old_name = next(iter(renames))
            self.category_status.set(f"Catégorie renommée : « {old_name} » → « {new_name} ».")
        else:
            merged = ", ".join(f"« {n} »" for n in renames)
            self.category_status.set(f"{len(renames)} catégories fusionnées et renommées en « {new_name} » : {merged}.")

    def _open_category_folder(self) -> None:
        if not self.selected_category_names or not self.dataset_dir:
            return
        for name in self.selected_category_names:
            folder = os.path.join(self.dataset_dir, name)
            if os.path.isdir(folder):
                os.startfile(folder)

    def _prefix_files(self) -> None:
        names = self.selected_category_names
        if not names or not self.dataset_dir:
            return
        total = sum(rename_files_with_prefix(self.base_model_path.get(), name) for name in names)
        self._on_select_category()  # rafraîchit la liste des fichiers affichés
        if total:
            self.category_status.set(f"{total} fichier(s) préfixé(s).")
        else:
            self.category_status.set("Tous les fichiers étaient déjà préfixés.")

    def _move_selected_files(self) -> None:
        """Déplace les fichiers sélectionnés (sélection multiple possible,
        Ctrl/Shift+clic) de la catégorie actuelle vers celle choisie dans le
        champ à côté du bouton "Déplacer" — sans passer par un renommage de
        catégorie entière, pour ne bouger que les documents concernés.
        Nécessite exactement UNE catégorie source sélectionnée (le bouton
        est désactivé sinon — voir `_on_select_category`)."""
        if len(self.selected_category_names) != 1:
            return
        name = self.selected_category_names[0]
        selected_indices = self.category_files_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Aucune sélection", "Sélectionnez d'abord un ou plusieurs fichiers dans la liste.")
            return
        target = self.move_target_var.get().strip()
        if not target:
            messagebox.showwarning("Catégorie manquante", "Indiquez la catégorie de destination.")
            return
        if target == name:
            messagebox.showinfo("Sans effet", "La catégorie de destination est déjà la catégorie actuelle.")
            return

        filenames = [self.category_files_listbox.get(i) for i in selected_indices]
        try:
            moved = move_files_to_category(self.base_model_path.get(), filenames, name, target)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = model_store.load_bundle(self.base_model_path.get())
        self._populate_categories()
        self.category_status.set(f"{moved} fichier(s) déplacé(s) de « {name} » vers « {target} ».")

    def _add_files_to_selected_category(self) -> None:
        """Sens inverse de "Déplacer" : au lieu de partir d'un fichier déjà
        classé pour lui choisir sa catégorie, choisir des fichiers n'importe
        où sur le disque et les affecter directement à la catégorie
        actuellement sélectionnée (une seule), sans passer par la
        prédiction de l'onglet Classification."""
        if len(self.selected_category_names) != 1:
            return
        category = self.selected_category_names[0]
        paths = filedialog.askopenfilenames(title=f"Ajouter des fichiers à « {category} »", filetypes=FILETYPES_DOCS)
        if not paths:
            return
        try:
            added = add_files_to_category(self.base_model_path.get(), list(paths), category)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = model_store.load_bundle(self.base_model_path.get())
        self.dataset_dir = model_store.model_dataset_dir(self.base_model_path.get())
        self._populate_categories()
        self.category_status.set(f"{added} fichier(s) ajouté(s) directement à « {category} ».")

    def _delete_category_selected(self) -> None:
        names = self.selected_category_names
        if not names:
            return
        other_name = get_config().other_category_name
        names_to_delete = [n for n in names if n != other_name]
        if not names_to_delete:
            return
        label = f"« {names_to_delete[0]} »" if len(names_to_delete) == 1 else f"{len(names_to_delete)} catégories ({', '.join(names_to_delete)})"
        if not messagebox.askyesno(
            "Confirmer",
            f"Supprimer {label} ?\n\n"
            f"Ses documents seront regroupés dans « {other_name} » — rien n'est perdu.",
        ):
            return
        try:
            bundle = rename_categories(self.base_model_path.get(), {n: other_name for n in names_to_delete})
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        self.bundle = bundle
        self._populate_categories()
        self.category_status.set(f"{len(names_to_delete)} catégorie(s) supprimée(s) (fusionnée(s) dans « {other_name} »).")

    def _delete_category_permanently_selected(self) -> None:
        """Contrairement à `_delete_category_selected` ci-dessus, ne fusionne
        PAS les documents dans "autre" : leurs copies `dataset/` sont
        perdues (les fichiers SOURCE, eux, ne sont jamais touchés — voir
        `rename.delete_category_permanently`)."""
        names = self.selected_category_names
        if not names:
            return
        label = f"« {names[0]} »" if len(names) == 1 else f"{len(names)} catégories ({', '.join(names)})"
        if not messagebox.askyesno(
            "Confirmer la suppression définitive",
            f"Supprimer définitivement {label} ?\n\n"
            "Contrairement à \"Supprimer la sélection (→ autre)\", ceci ne fusionne PAS les "
            "documents ailleurs : leurs copies dans le dossier dataset/ du modèle seront "
            "perdues (un instantané est pris avant, voir \"Historique\" pour annuler). Les "
            "fichiers d'origine, eux, ne sont jamais touchés.",
        ):
            return
        model_path = self.base_model_path.get()
        total_removed = 0
        try:
            for name in names:
                total_removed += delete_category_permanently(model_path, name)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
        finally:
            self.bundle = model_store.load_bundle(model_path)
            self._populate_categories()
        self.category_status.set(f"{len(names)} catégorie(s) supprimée(s) définitivement ({total_removed} fichier(s)).")

    def _delete_selected_files(self) -> None:
        """Supprime DÉFINITIVEMENT les copies `dataset/` des fichiers
        sélectionnés (sélection multiple possible), sans les faire passer
        par "autre" — voir `rename.delete_file_from_category`. Nécessite
        exactement UNE catégorie source sélectionnée, comme "Déplacer"."""
        if len(self.selected_category_names) != 1:
            return
        name = self.selected_category_names[0]
        selected_indices = self.category_files_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Aucune sélection", "Sélectionnez d'abord un ou plusieurs fichiers dans la liste.")
            return
        filenames = [self.category_files_listbox.get(i) for i in selected_indices]
        label = f"« {filenames[0]} »" if len(filenames) == 1 else f"{len(filenames)} fichiers"
        if not messagebox.askyesno(
            "Confirmer la suppression définitive",
            f"Supprimer définitivement {label} de « {name} » ?\n\n"
            "Seule la copie dans le dossier dataset/ du modèle est supprimée (un instantané "
            "est pris avant, voir \"Historique\" pour annuler) — le fichier d'origine, lui, "
            "n'est jamais touché.",
        ):
            return
        model_path = self.base_model_path.get()
        removed = 0
        try:
            for filename in filenames:
                if delete_file_from_category(model_path, name, filename):
                    removed += 1
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
        finally:
            self.bundle = model_store.load_bundle(model_path)
            self._populate_categories()
            self._on_select_category()
        self.category_status.set(f"{removed} fichier(s) supprimé(s) définitivement de « {name} ».")

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
        extensions = self._selected_extensions()
        if not extensions:
            messagebox.showwarning(
                "Aucun type de fichier sélectionné",
                "Cochez au moins un type de fichier à inclure dans l'entraînement.",
            )
            return
        model_path = model_store.model_path_for_name(model_name)

        self.train_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        # Désactivé pendant l'entraînement pour éviter de revectoriser
        # dataset/ pendant qu'il est en train d'être réécrit par ce même
        # entraînement (voir `_show_cluster_preview`) — réactivé ensuite par
        # `_populate_categories` (déclenché après coup si ce modèle est
        # celui sélectionné comme base à améliorer, voir `_on_model_trained`).
        self.cluster_preview_button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress_bar.start(12)

        engine_name = self.engine.get()
        if engine_name == ENGINE_EMBEDDINGS:
            embedding_model = self.embedding_model.get().strip()
        elif engine_name == ENGINE_IMAGE:
            embedding_model = self.image_model.get().strip()
        else:
            embedding_model = None
        base_model_path = self.base_model_path.get().strip() or None
        overrides = {
            "k_min": self.k_min_var.get(),
            "k_max": self.k_max_var.get(),
            "min_silhouette": self.min_silhouette_var.get(),
            "min_cluster_size": self.min_cluster_size_var.get(),
            "tfidf_max_features": self.tfidf_max_features_var.get(),
            "tfidf_ngram_max": self.tfidf_ngram_max_var.get(),
            "extensions": extensions,
            "recursive": self.recursive_var.get(),
            "use_keybert": self.use_keybert_var.get(),
            "detect_duplicates": self.detect_duplicates_var.get(),
            "tfidf_use_stemming": self.tfidf_use_stemming_var.get(),
            "cluster_algorithm": self.cluster_algorithm_var.get(),
            "cluster_use_svd": self.cluster_use_svd_var.get(),
            "cluster_svd_components": self.cluster_svd_components_var.get(),
            "cluster_extra_stopwords": self.cluster_extra_stopwords_var.get(),
            "image_cluster_labels": self.image_cluster_labels_var.get(),
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
        # Un dossier source devenu introuvable (ex. un ancien modèle de
        # chaînage déplacé ou supprimé depuis) ne fait qu'un avertissement
        # discret dans le journal, vite noyé parmi les nombreuses lignes de
        # progression — d'où sa remontée explicite en fin d'entraînement,
        # plutôt que de laisser l'utilisateur découvrir plus tard que des
        # catégories entières ont disparu sans explication.
        warnings: list[str] = []

        def progress(message: str) -> None:
            self.after(0, self._log_line, message)
            if message.startswith("⚠") and "introuvable" in message:
                warnings.append(message)

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
            self.after(0, self._check_training_duplicates, model_path)
            if warnings:
                self.after(0, lambda: messagebox.showwarning(
                    "Modèle entraîné — avec avertissement(s)",
                    "Le modèle a bien été (ré)entraîné, mais :\n\n" + "\n".join(warnings) + "\n\n"
                    "Un dossier source manquant (ex. un ancien modèle déplacé ou supprimé) fait "
                    "perdre au ré-entraînement l'accès aux documents qu'il contenait : les "
                    "catégories correspondantes peuvent être devenues plus petites, voire avoir "
                    "disparu.",
                ))
        except Exception as exc:
            self.after(0, self._log_line, f"\nErreur : {exc}")
        finally:
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.train_button.configure(state="normal"))

    def _show_cluster_preview(self) -> None:
        """Revectorise à la demande (voir `rename.compute_dataset_vectors`)
        tous les fichiers actuellement présents dans `dataset/<catégorie>/`
        du modèle chargé ci-dessus ("Modèle existant à améliorer"), plutôt
        que de dépendre d'un instantané pris pendant un entraînement — ça
        marche donc pour n'importe quel modèle non supervisé déjà entraîné,
        qu'on vienne de relancer son entraînement ou simplement de le
        sélectionner, et reflète toujours l'état ACTUEL de ses catégories
        (renommages compris, voir `list_category_files`/`_populate_categories`,
        même source de données)."""
        if not self.bundle or self.bundle.get("mode") != "unsupervised":
            messagebox.showinfo(
                "Aperçu du regroupement",
                "Sélectionnez d'abord, dans la section \"Modèle existant à améliorer\" "
                "ci-dessus, un modèle à catégories détectées automatiquement.",
            )
            return
        if not self.dataset_dir or not os.path.isdir(self.dataset_dir):
            messagebox.showinfo(
                "Aperçu du regroupement",
                "Ce modèle n'a pas encore de dossier dataset/ — entraînez-le ou "
                "améliorez-le une première fois.",
            )
            return

        bundle, dataset_dir = self.bundle, self.dataset_dir
        self.cluster_preview_button.configure(state="disabled", text="Aperçu du regroupement (calcul en cours...)")
        self.progress_bar.start(12)

        def progress(message: str) -> None:
            self.after(0, self._log_line, message)

        def worker() -> None:
            try:
                vectors, labels = compute_dataset_vectors_fn(bundle, dataset_dir, progress=progress)
            except Exception as exc:
                self.after(0, self._cluster_preview_failed, str(exc))
                return
            self.after(0, self._cluster_preview_ready, vectors, labels)

        threading.Thread(target=worker, daemon=True).start()

    def _cluster_preview_failed(self, message: str) -> None:
        self._reset_cluster_preview_button()
        messagebox.showerror("Erreur", f"Impossible de calculer l'aperçu du regroupement :\n{message}")

    def _cluster_preview_ready(self, vectors, labels: list[str]) -> None:
        self._reset_cluster_preview_button()
        if not labels:
            messagebox.showinfo(
                "Aperçu du regroupement", "Aucun document lisible trouvé dans le dossier dataset/ de ce modèle.",
            )
            return
        ClusterPreviewWindow(self, vectors, labels)

    def _reset_cluster_preview_button(self) -> None:
        self.progress_bar.stop()
        self.cluster_preview_button.configure(state="normal", text="Aperçu du regroupement (2D)")

    def _enable_open_folder(self, model_path: str) -> None:
        self.last_model_path = model_path
        self.open_folder_button.configure(state="normal")
        # Le modèle tout juste (ré)entraîné doit apparaître immédiatement
        # dans "Modèles disponibles" ci-dessus, sans avoir à cliquer sur
        # "Rafraîchir" à la main pour pouvoir l'améliorer à nouveau ensuite.
        self._refresh_base_model_picker()

    def _open_preview_dir(self) -> None:
        if not self.last_model_path:
            return
        dataset_dir = model_store.model_dataset_dir(self.last_model_path)
        if os.path.isdir(dataset_dir):
            os.startfile(dataset_dir)

    def _check_training_duplicates(self, model_path: str) -> None:
        """Propose de déplacer les documents en double vers un dossier
        _backup si l'entraînement en a détecté (voir la case "Détecter les
        documents en double" dans les paramètres avancés) — enregistrés dans
        le manifest du modèle (`<nom>.json`), pas dans un fichier séparé."""
        manifest = model_store.load_manifest(model_path)
        pairs = manifest.get("duplicates") or []
        if not pairs:
            return
        manifest_path = model_store.model_manifest_path(model_path)
        DuplicatesDialog(
            self, pairs, on_confirm=lambda: self._delete_training_duplicates(manifest_path),
            note=(
                "Un exemplaire de chaque groupe de doublons est gardé. Les documents d'origine ne "
                "sont jamais touchés : seule leur COPIE dans le dossier du modèle (dataset/) est "
                "déplacée vers dataset/_backup/<catégorie>/ — un seul dossier de secours à la racine "
                "du dataset, jamais une suppression définitive, toujours récupérable au besoin."
            ),
        )

    def _delete_training_duplicates(self, manifest_path: str) -> None:
        def task() -> None:
            moved = delete_training_duplicates_fn(manifest_path, progress=lambda m: self.after(0, self._log_line, m))
            if moved:
                self.after(
                    0, lambda: messagebox.showinfo(
                        "Doublons déplacés",
                        f"{len(moved)} document(s) déplacé(s) vers dataset/_backup/ "
                        "(un seul dossier de secours, avec un sous-dossier par catégorie).",
                    )
                )

        threading.Thread(target=task, daemon=True).start()


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


class AutomationTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self.body = scrollable.body
        self.manager = AutomationManager(on_event=self._on_event)
        self._build()
        try:
            self.manager.load()
        except Exception as exc:
            # Filet de sécurité : `AutomationManager.load()` est déjà
            # tolérante ligne par ligne, mais une erreur totalement
            # inattendue ici ne doit dans tous les cas jamais empêcher le
            # reste de l'application de démarrer.
            self._on_event(f"⚠ Chargement des automatisations impossible : {exc}")
        self._schedule_refresh()

    def _build(self) -> None:
        buttons = ttk.Frame(self.body)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Ajouter...", command=self._add).pack(side="left")
        ttk.Button(buttons, text="Modifier...", command=self._edit).pack(side="left", padx=6)
        ttk.Button(buttons, text="Supprimer", command=self._delete).pack(side="left")
        ttk.Button(buttons, text="Démarrer", command=self._start_selected).pack(side="left", padx=(20, 6))
        ttk.Button(buttons, text="Arrêter", command=self._stop_selected).pack(side="left")

        columns = ("name", "watch", "model", "interval", "status", "last_run", "count")
        self.tree = ttk.Treeview(self.body, columns=columns, show="headings", height=8)
        headers = {
            "name": "Nom", "watch": "Dossier surveillé", "model": "Modèle", "interval": "Intervalle",
            "status": "Statut", "last_run": "Dernier passage", "count": "Fichiers",
        }
        widths = {"name": 120, "watch": 220, "model": 150, "interval": 90, "status": 70, "last_run": 150, "count": 65}
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths[col])
        self.tree.pack(fill="x", pady=10)

        ttk.Label(self.body, text="Journal :").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(self.body, height=14, state="disabled")
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
        super().__init__(parent)
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self.body = scrollable.body
        self.server: api_server.ApiServer | None = None
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self.body)
        top.pack(fill="x")
        ttk.Label(top, text="Port :").pack(side="left")
        self.port_var = tk.IntVar(value=get_config().api_port)
        ttk.Spinbox(top, from_=1024, to=65535, textvariable=self.port_var, width=8).pack(side="left", padx=6)
        self.toggle_button = ttk.Button(top, text="Démarrer le serveur API", command=self._toggle_server)
        self.toggle_button.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Serveur arrêté.")
        ttk.Label(self.body, textvariable=self.status_var, foreground="#555").pack(anchor="w", pady=(8, 0))

        key_row = ttk.Frame(self.body)
        key_row.pack(fill="x", pady=(4, 0))
        ttk.Label(key_row, text="Clé API (en-tête \"Authorization: Bearer <clé>\") :").pack(side="left")
        self.token_var = tk.StringVar(value="(démarrez le serveur pour générer une clé)")
        ttk.Entry(key_row, textvariable=self.token_var, width=38, state="readonly").pack(side="left", padx=6)
        ttk.Button(key_row, text="Copier", command=self._copy_token).pack(side="left")

        warning = tk.Label(
            self.body,
            text="N'écoute que sur cette machine (127.0.0.1), jamais sur le réseau. "
            "Tout programme lancé sur cette machine et connaissant la clé peut piloter l'application : "
            "ne la partagez pas, et arrêtez le serveur quand vous n'en avez plus besoin.",
            justify="left", wraplength=760, anchor="w", bg="#fff3cd", fg="#664d03", padx=8, pady=6,
        )
        warning.pack(fill="x", pady=(8, 0))

        ttk.Label(self.body, text="Routes disponibles :", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(14, 4)
        )
        self.doc_text = scrolledtext.ScrolledText(self.body, height=14, state="disabled", wrap="word")
        self.doc_text.pack(fill="both", expand=True)
        self._fill_docs()

        ttk.Label(self.body, text="Journal des requêtes :").pack(anchor="w", pady=(10, 2))
        self.log = scrolledtext.ScrolledText(self.body, height=8, state="disabled")
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
# Couleur d'avertissement (orange) réutilisée pour le bouton "?" ET la
# bannière de la modale des sections dont une partie ne fonctionne pas du
# tout sans un paquet optionnel supplémentaire (au-delà du simple confort
# habituel des paquets optionnels — voir `SETTINGS_PACKAGE_REQUIREMENTS`).
_PACKAGE_WARNING_COLOR = "#d35400"
_PACKAGE_WARNING_BG = "#fdf0e3"

# Sections de l'onglet Paramètres dont au moins un réglage reste inopérant
# (erreur explicite, pas un simple repli silencieux) tant qu'un paquet
# optionnel — et, pour l'OCR, un exécutable externe — n'est pas installé.
# Affiché en orange (bouton "?" ET bannière de la modale, voir
# `_add_help_button`/`FieldHelpDialog`) pour que ce ne soit pas découvert
# seulement au premier essai raté. Volontairement limité aux cas où c'est
# une vraie erreur bloquante : le repli automatique de la racinisation
# (`tfidf_use_stemming`, `snowballstemmer` absent) ou de KeyBERT
# (`cluster_use_keybert`, paquet absent) ne casse rien, donc pas de bannière
# pour la seule section "clustering".
SETTINGS_PACKAGE_REQUIREMENTS: dict[str, dict] = {
    "vectorization": {
        "note": (
            "Les moteurs \"embeddings\" ET \"image\" (analyse visuelle CLIP) ne fonctionnent "
            "pas du tout sans le paquet optionnel sentence-transformers (erreur explicite au "
            "lancement de l'entraînement, pas un repli silencieux) — TF-IDF, lui, n'a besoin "
            "d'aucun paquet supplémentaire."
        ),
        "links": [
            ("requirements-embeddings.txt (installation)", "https://pypi.org/project/sentence-transformers/"),
            ("sentence-transformers — documentation", "https://www.sbert.net/"),
        ],
    },
    "ocr": {
        "note": (
            "Le paquet Python pytesseract est installé par défaut (requirements.txt) — mais "
            "l'OCR ne fonctionne quand même pas sans le moteur Tesseract lui-même, un "
            "exécutable externe (PAS un paquet pip) à installer séparément sur la machine. "
            "Sans lui, un PDF scanné ou une image reste \"illisible\" comme si l'OCR "
            "n'existait pas."
        ),
        "links": [
            ("Tesseract — guide d'installation (toutes plateformes)", "https://tesseract-ocr.github.io/tessdoc/Installation.html"),
            ("Tesseract pour Windows (installeur UB-Mannheim)", "https://github.com/UB-Mannheim/tesseract/wiki"),
        ],
    },
}

# Explication détaillée de chaque réglage technique, groupée par section
# (même regroupement que les LabelFrame de SettingsTab) — affichée dans une
# modale dédiée (bouton "?" en haut à droite de chaque section) plutôt que
# comme texte permanent, pour garder le formulaire principal lisible.
SETTINGS_HELP: dict[str, list[tuple[str, str]]] = {
    "clustering": [
        (
            "Nombre minimal de catégories à essayer",
            "Borne basse du nombre de catégories (k) que le regroupement automatique "
            "(K-Means) essaie de former. Le meilleur k entre ce minimum et le maximum "
            "ci-dessous est choisi automatiquement via le score de silhouette.",
        ),
        (
            "Nombre maximal de catégories à essayer",
            "Borne haute du nombre de catégories essayées. Aucun autre plafond n'est "
            "appliqué dans le code : augmentez librement cette valeur si vous avez besoin "
            "de plus de catégories — la seule autre limite réelle est le nombre de "
            "documents lisibles (on ne peut pas former plus de groupes que de documents).",
        ),
        (
            "Mots-clés utilisés pour nommer une catégorie",
            "Nombre de mots-clés assemblés pour proposer un nom de catégorie détectée "
            "(ex. 3 mots-clés → \"facture_tva_fournisseur\"). Utilisé en repli TF-IDF si "
            "KeyBERT est désactivé ou indisponible, et comme nombre de mots-clés/groupes "
            "de mots demandés à KeyBERT sinon.",
        ),
        (
            "Score de silhouette minimal pour découper",
            "Score de silhouette minimal (entre -1 et 1) pour accepter un découpage en "
            "plusieurs catégories. En dessous de ce seuil pour TOUTES les valeurs de k "
            "testées, le regroupement renonce à découper et garde une seule catégorie.\n\n"
            "ATTENTION — ce score n'est PAS comparable d'un moteur à l'autre : mesuré sur "
            "un vrai jeu de test à 3 catégories réellement distinctes, TF-IDF n'atteint "
            "qu'environ 0.03 pour le DÉCOUPAGE CORRECT, là où les embeddings atteignent "
            "0.19 pour ce même découpage — TF-IDF opère dans un espace très creux et de "
            "grande dimension où le score est structurellement bien plus bas, même quand "
            "la séparation est réelle. La valeur par défaut est donc un compromis "
            "délibérément permissif pensé pour ne pas bloquer une vraie séparation avec "
            "TF-IDF, pas une garantie absolue : \"Documents minimum par catégorie\" "
            "ci-dessous reste la protection la plus fiable contre un découpage dégénéré. "
            "Les préréglages de l'onglet Entraînement permettent d'ajuster ce compromis "
            "au cas par cas plutôt que de dépendre d'une seule valeur globale.",
        ),
        (
            "Documents minimum par catégorie",
            "Nombre minimal de documents pour qu'une catégorie soit retenue lors de la "
            "recherche automatique de k. Sans ce plancher, le score de silhouette grimpe "
            "artificiellement à mesure que k se rapproche du nombre total de documents : "
            "un cluster réduit à un seul document a toujours une cohésion parfaite (aucune "
            "distance interne), ce qui gonfle son score sans refléter une vraie catégorie "
            "— mesuré sur un petit corpus où le score \"optimal\" grimpait jusqu'à k = "
            "presque tous les documents, produisant une catégorie par document ou presque "
            "plutôt qu'une catégorisation utilisable.",
        ),
        (
            "Noms de catégorie plus naturels via KeyBERT",
            "Utilise KeyBERT (mots-clés choisis par similarité sémantique via un modèle "
            "d'embeddings, plutôt que par simple fréquence TF-IDF) pour nommer les "
            "catégories détectées — noms généralement plus naturels et lisibles. Réutilise "
            "all-MiniLM-L6-v2, déjà le modèle d'embeddings léger par défaut de "
            "l'application (aucun téléchargement supplémentaire). Se replie "
            "automatiquement sur le nommage TF-IDF si le paquet optionnel `keybert` n'est "
            "pas installé (`pip install -r requirements-embeddings.txt`).",
        ),
        (
            "Détecter les documents en double pendant l'entraînement",
            "Signale, pendant l'entraînement, les paires de documents quasi identiques "
            "(similarité cosinus ≥ 97 % sur les vecteurs déjà calculés pour le "
            "regroupement — aucun calcul supplémentaire lourd). Désactivé automatiquement "
            "au-delà de 4000 documents pour éviter un coût en O(n²) trop long sur un très "
            "gros corpus. Si des doublons sont trouvés, une fenêtre les liste à la fin de "
            "l'entraînement et propose de les déplacer vers dataset/_backup/<catégorie>/ — "
            "jamais une suppression définitive.",
        ),
        (
            "Case \"Réécrire la configuration\" cochée par défaut",
            "État par défaut, dans l'onglet Entraînement, de la case \"Réécrire la "
            "configuration\" (section \"Modèle existant à améliorer\"). Cochée, elle "
            "recharge automatiquement dans le formulaire les réglages EXACTS (moteur, "
            "k min/max, types de fichiers, sous-dossiers...) que le modèle sélectionné "
            "avait lors de son dernier entraînement, pour reprendre son amélioration dans "
            "les mêmes conditions plutôt que d'écraser silencieusement sa configuration "
            "avec ce qui restait dans le formulaire.",
        ),
        (
            "Algorithme de regroupement",
            "\"kmeans\" (défaut) recherche automatiquement le meilleur k entre les bornes "
            "min/max ci-dessus. \"minibatch_kmeans\" est une variante approximative bien "
            "plus rapide sur un gros corpus (le regroupement bascule dessus automatiquement "
            "au-delà du seuil ci-dessous, même si \"kmeans\" est sélectionné ici). "
            "\"agglomerative\" (hiérarchique, linkage de Ward) donne souvent des groupes "
            "plus cohérents sur un corpus petit à moyen, mais est plus lent. \"hdbscan\" ne "
            "force PAS chaque document dans un cluster : les documents trop atypiques "
            "restent \"non classés\" (fusionnés dans la catégorie \"autre\") plutôt que "
            "rattachés arbitrairement au cluster le moins pire, mais ne respecte pas les "
            "bornes min/max (le nombre de groupes est déduit des données).",
        ),
        (
            "Bascule kmeans → minibatch_kmeans",
            "Au-delà de ce nombre de documents lisibles, l'algorithme \"kmeans\" bascule "
            "automatiquement sur \"minibatch_kmeans\" (résultat très proche, beaucoup plus "
            "rapide) — sans ça, un entraînement sur un très gros dossier peut devenir "
            "extrêmement lent (KMeans plein recalcule sur tout le corpus à chaque "
            "itération). Les autres algorithmes ne sont pas concernés.",
        ),
        (
            "Réduire la dimension des vecteurs TF-IDF avant regroupement (SVD)",
            "Réduit la dimension des vecteurs TF-IDF (SVD tronquée, \"LSA\") avant le "
            "regroupement — l'espace TF-IDF est très creux et de grande dimension, ce qui "
            "structurellement écrase le score de silhouette même pour une séparation "
            "réelle ; une projection sur moins de dimensions resserre les distances et peut "
            "améliorer la qualité du découpage. Sans effet pour le moteur embeddings, déjà "
            "dense et de dimension raisonnable.",
        ),
        (
            "Composantes SVD",
            "Nombre de dimensions conservées par la réduction SVD ci-dessus (ignoré si "
            "désactivée, ou automatiquement réduit si le corpus ou le vocabulaire sont plus "
            "petits que cette valeur).",
        ),
        (
            "Afficher les métriques Davies-Bouldin/Calinski-Harabasz",
            "Calcule, en plus du score de silhouette déjà utilisé pour choisir le nombre de "
            "catégories, deux métriques complémentaires (Davies-Bouldin : plus bas est "
            "meilleur ; Calinski-Harabasz : plus haut est meilleur), affichées dans le "
            "journal d'entraînement — utile pour croiser plusieurs indices plutôt que "
            "dépendre d'un seul.",
        ),
        (
            "Mots à ignorer en plus",
            "Mots à exclure, en plus des mots vides français/anglais habituels, du nommage "
            "des catégories ET de la vectorisation TF-IDF (ex. le nom de l'entreprise, qui "
            "apparaît dans chaque document et pollue sinon le nommage) — séparés par des "
            "virgules. Valeur de départ du formulaire de l'onglet Entraînement, modifiable "
            "pour un entraînement précis sans changer ce défaut.",
        ),
    ],
    "vectorization": [
        (
            "Vocabulaire TF-IDF maximal",
            "Nombre maximal de mots/groupes de mots distincts retenus par le moteur "
            "TF-IDF pour représenter les documents. Une valeur plus grande capture un "
            "vocabulaire plus riche (utile sur un corpus varié) au prix d'un calcul un "
            "peu plus lourd ; une valeur plus petite se concentre sur les termes les plus "
            "fréquents. Sans effet sur le moteur embeddings.",
        ),
        (
            "Taille maximale des groupes de mots (n-grammes)",
            "Longueur maximale des groupes de mots consécutifs considérés comme une seule "
            "unité par TF-IDF (1 = mots isolés seulement, 2 = mots isolés + paires de "
            "mots comme \"virement bancaire\", etc.). Une valeur plus grande capture mieux "
            "les expressions figées, au prix d'un vocabulaire plus grand. Sans effet sur "
            "le moteur embeddings.",
        ),
        (
            "Modèle d'embeddings par défaut",
            "Modèle de langage pré-entraîné utilisé quand le moteur \"embeddings\" est "
            "choisi dans l'onglet Entraînement, du plus léger au plus lourd (voir le "
            "bouton \"? Explications\" de cet onglet pour le détail de chaque modèle). "
            "Téléchargé automatiquement au tout premier usage, puis réutilisé.",
        ),
        (
            "Racinisation des mots (FR/EN) avant la vectorisation TF-IDF",
            "Racinise les mots (stemming, ex. \"facture\"/\"factures\"/\"facturé\" -> "
            "\"factur\") avant de les compter pour le TF-IDF, pour que des variantes d'un "
            "même mot ne soient plus des jetons distincts qui diluent leur poids respectif. "
            "Se replie automatiquement sur les mots tels quels si le paquet optionnel "
            "`snowballstemmer` n'est pas installé (`pip install -r requirements-nlp.txt`). "
            "Sans effet sur le moteur embeddings. Les noms de catégorie affichés restent "
            "toujours de vrais mots, jamais une racine tronquée.",
        ),
        (
            "Utiliser le GPU (CUDA) pour le moteur embeddings",
            "Utilise le GPU (CUDA) pour le moteur embeddings s'il en détecte un disponible "
            "— nettement plus rapide sur un gros corpus. Sans effet (et sans erreur) si "
            "aucun GPU compatible n'est détecté : repli silencieux sur le CPU. S'applique "
            "aussi au moteur \"image\" ci-dessous (même mécanisme de détection GPU).",
        ),
        (
            "Modèle CLIP par défaut (moteur image)",
            "Modèle utilisé par le moteur \"image\" de l'onglet Entraînement, qui catégorise "
            "des photos par leur contenu VISUEL (personnes, scènes, objets) plutôt que par un "
            "texte extrait — capacité différente de l'OCR (qui LIT le texte d'une image sans "
            "rien comprendre à la scène). \"clip-ViT-B-32\" (léger, ~350 Mo) est recommandé "
            "pour démarrer ; \"clip-ViT-L-14\" (~890 Mo) est plus précis mais plus lent, utile "
            "pour départager finement beaucoup de photos visuellement proches.",
        ),
        (
            "Libellés candidats (moteur image)",
            "Le moteur \"image\" n'a pas de texte propre au document dont extraire des "
            "mots-clés (contrairement à TF-IDF/KeyBERT) : chaque catégorie détectée est plutôt "
            "nommée par comparaison à cette liste de libellés candidats (\"personnes\", "
            "\"paysage\", \"document texte\"...) — le libellé le plus proche du contenu visuel "
            "moyen du groupe devient son nom (nommage \"zero-shot\" via CLIP, sans exemple "
            "d'entraînement nécessaire pour ces libellés). Modifiable pour un entraînement "
            "précis (comme les mots à ignorer du regroupement TF-IDF) sans changer ce défaut.",
        ),
    ],
    "classification": [
        (
            "Seuil de confiance minimal (0 à 1)",
            "En dessous de ce niveau de confiance, une prédiction est placée dans la "
            "catégorie \"incertain\" plutôt que d'être classée en silence avec un risque "
            "d'erreur — mieux vaut vous demander de vérifier à la main qu'une catégorie "
            "peu fiable proposée comme si elle était sûre.",
        ),
        (
            "Nom de la catégorie \"incertain\"",
            "Nom du dossier/catégorie où sont rangés les documents dont la prédiction est "
            "sous le seuil de confiance minimal ci-dessus.",
        ),
        (
            "Nom de la catégorie \"illisible\"",
            "Nom du dossier/catégorie où sont rangés les documents dont aucun texte n'a pu "
            "être extrait (PDF scanné sans OCR, fichier corrompu, format non pris en "
            "charge...). Ces documents peuvent être recatégorisés manuellement dans "
            "l'onglet Classification, mais ne peuvent jamais contribuer à l'amélioration "
            "du modèle faute de texte à analyser.",
        ),
        (
            "Nom de la catégorie \"autre\" (suppression)",
            "Catégorie fourre-tout où sont regroupés les documents d'une catégorie "
            "supprimée dans la section \"Catégories de ce modèle\" de l'onglet Entraînement — "
            "les documents ne sont jamais perdus, seulement déplacés ici.",
        ),
        (
            "Case \"Améliorer le modèle\" cochée par défaut",
            "État par défaut, dans l'onglet Classification, de la case \"Améliorer le "
            "modèle avec ces documents (catégories corrigées prises comme référence)\". "
            "Cochée, chaque validation où au moins un document a été corrigé à la main "
            "relance automatiquement l'entraînement du modèle chargé avec ces corrections.",
        ),
        (
            "Case \"Inclure les fichiers à vérifier dans l'export\" cochée par défaut",
            "État par défaut, dans l'onglet Classification, de la case qui inclut (ou non) "
            "les documents classés \"incertain\" dans l'export final. Décochée, ces "
            "fichiers restent simplement à leur emplacement d'origine plutôt que d'être "
            "copiés/déplacés.",
        ),
        (
            "Case \"Inclure les fichiers non catégorisé dans l'export\" cochée par défaut",
            "Comme ci-dessus, mais pour les documents classés \"illisible\" (aucun texte "
            "extrait).",
        ),
    ],
    "folders": [
        (
            "Dossier racine des modèles",
            "Dossier où chaque modèle créé depuis l'onglet Entraînement obtient son "
            "propre sous-dossier : <racine>/<nom>/<nom>.pkl, <nom>.json (fichiers "
            "référencés) et dataset/<catégorie>/ (documents utilisés pour le construire "
            "et l'améliorer). Toujours accessible même après un redémarrage de "
            "l'application, sans dépendre d'un dossier temporaire.",
        ),
        (
            "Dossier de sortie par défaut (Classification)",
            "Dossier proposé par défaut dans l'onglet Classification pour l'export des "
            "documents classés — modifiable au cas par cas via le champ \"Dossier de "
            "sortie\" de cet onglet.",
        ),
    ],
    "automation": [
        (
            "Fichier de configuration des automatisations",
            "Fichier JSON où sont enregistrées les automatisations créées dans l'onglet "
            "Automatisation (nom, dossier surveillé, modèle, intervalle...), pour les "
            "retrouver au redémarrage de l'application.",
        ),
        (
            "Intervalle par défaut (valeur)",
            "Valeur numérique proposée par défaut pour l'intervalle entre deux passages "
            "d'une nouvelle automatisation (à combiner avec l'unité ci-dessous, ex. "
            "\"10 minutes\").",
        ),
        (
            "Intervalle par défaut (unité)",
            "Unité de temps proposée par défaut pour l'intervalle d'une nouvelle "
            "automatisation (minutes, heures ou jours).",
        ),
        (
            "Déplacer les fichiers par défaut",
            "État par défaut, pour une nouvelle automatisation, de l'option qui déplace "
            "les fichiers traités (au lieu de les copier) — les fichiers d'origine "
            "disparaissent alors du dossier surveillé une fois classés.",
        ),
        (
            "Inclure les fichiers \"à vérifier\" par défaut",
            "État par défaut, pour une nouvelle automatisation, de l'inclusion des "
            "documents classés \"incertain\" dans son export.",
        ),
        (
            "Inclure les fichiers \"non catégorisé\" par défaut",
            "État par défaut, pour une nouvelle automatisation, de l'inclusion des "
            "documents classés \"illisible\" (aucun texte extrait) dans son export.",
        ),
    ],
    "misc": [
        (
            "Profondeur de recherche des modèles .pkl",
            "Profondeur maximale de sous-dossiers explorée quand l'application recherche "
            "automatiquement les modèles `.pkl` déjà présents dans le projet (liste "
            "déroulante \"Modèles disponibles\" de l'onglet Classification). Les dossiers "
            "cachés et les dossiers internes (pkl_history, json_history, dataset_history, "
            "_backup...) sont toujours ignorés, quelle que soit cette profondeur.",
        ),
        (
            "Instantanés conservés par modèle",
            "Nombre d'instantanés (pkl_history/, json_history/, dataset_history/) "
            "conservés par modèle avant qu'un entraînement/amélioration/renommage "
            "ultérieur ne l'écrase — permet de revenir en arrière depuis \"Historique / "
            "Revenir en arrière...\" (onglet Classification). 0 = aucun "
            "historique conservé (aucun retour en arrière possible).",
        ),
        (
            "Largeur de fenêtre par défaut",
            "Largeur, en pixels, de la fenêtre de l'application à son lancement.",
        ),
        (
            "Hauteur de fenêtre par défaut",
            "Hauteur, en pixels, de la fenêtre de l'application à son lancement. Quelle "
            "que soit cette valeur, chaque onglet reste défilable si son contenu dépasse "
            "la hauteur réellement disponible.",
        ),
        (
            "Port du serveur API local",
            "Port réseau proposé par défaut pour le serveur API local (onglet API, "
            "bouton \"Démarrer le serveur API\") — modifiable au cas par cas dans cet "
            "onglet. Le serveur n'écoute jamais que sur 127.0.0.1 (jamais exposé au "
            "réseau) ; une clé aléatoire, régénérée à chaque démarrage, est requise sur "
            "toutes les routes sauf \"/\" et \"/health\".",
        ),
    ],
    "extraction": [
        (
            "Fichiers extraits en parallèle",
            "Nombre de fichiers dont le texte est extrait en même temps (threads) lors "
            "d'un scan de dossier. L'extraction délègue l'essentiel du travail à des "
            "bibliothèques qui relâchent le verrou global Python (pymupdf...) ou attend "
            "simplement des lectures disque : le parallélisme profite donc réellement, "
            "surtout sur un gros dossier. 1 désactive le parallélisme (extraction "
            "strictement séquentielle, utile pour un diagnostic).",
        ),
    ],
    "ocr": [
        (
            "Activer l'OCR (PDF scannés, images)",
            "Permet de lire le texte d'un PDF scanné (sans couche texte native — tenté "
            "seulement quand l'extraction normale ne donne rien) et des fichiers image "
            "(.png, .jpg, .jpeg, .tiff, .bmp) par reconnaissance de caractères (Tesseract). "
            "Le paquet Python pytesseract est installé par défaut (requirements.txt) ; seul "
            "le moteur Tesseract lui-même reste à installer séparément sur la machine (pas "
            "un paquet pip — voir le README pour les liens d'installation). Désactivé par "
            "défaut : un PDF scanné ou une image reste simplement \"illisible\" (comme "
            "aujourd'hui) tant que ce n'est pas activé.",
        ),
        (
            "Chemin de l'exécutable Tesseract",
            "Chemin complet vers l'exécutable Tesseract si celui-ci n'est pas sur le PATH "
            "du système (ex. sous Windows, l'installeur ne l'y ajoute pas toujours) — "
            "laissez vide s'il est déjà détecté automatiquement.",
        ),
    ],
}


class FieldHelpDialog(tk.Toplevel):
    """Explication détaillée de chaque réglage d'une section de l'onglet
    Paramètres, ouverte via le bouton "?" en haut à droite de cette section
    plutôt qu'affichée en permanence dans le formulaire (déjà dense).

    `requirement`, si fourni (voir `SETTINGS_PACKAGE_REQUIREMENTS`), affiche
    en plus une bannière orange en tête de modale : au moins un réglage de
    cette section reste inopérant (pas un simple repli silencieux) tant
    qu'un paquet optionnel n'est pas installé — avec les liens de
    téléchargement/installation correspondants, cliquables directement."""

    def __init__(
        self, parent: tk.Widget, section_title: str, entries: list[tuple[str, str]],
        requirement: dict | None = None,
    ):
        super().__init__(parent)
        self.title(f"{section_title} — explications")
        self.geometry("640x560")
        self.transient(parent.winfo_toplevel())

        text = scrolledtext.ScrolledText(self, wrap="word", padx=12, pady=10)
        text.pack(fill="both", expand=True)
        text.tag_configure("h1", font=("TkDefaultFont", 12, "bold"), spacing1=4, spacing3=6)
        text.tag_configure("h2", font=("TkDefaultFont", 10, "bold"), spacing1=10, spacing3=2)
        text.tag_configure("body", font=("TkDefaultFont", 9), spacing3=4)
        text.tag_configure(
            "warning_title", font=("TkDefaultFont", 9, "bold"),
            foreground=_PACKAGE_WARNING_COLOR, background=_PACKAGE_WARNING_BG,
            spacing1=8, spacing3=2, lmargin1=6, lmargin2=6,
        )
        text.tag_configure(
            "warning_body", font=("TkDefaultFont", 9),
            foreground=_PACKAGE_WARNING_COLOR, background=_PACKAGE_WARNING_BG,
            spacing3=2, lmargin1=6, lmargin2=6,
        )
        text.tag_configure(
            "warning_link", font=("TkDefaultFont", 9, "underline"),
            foreground=_PACKAGE_WARNING_COLOR, background=_PACKAGE_WARNING_BG,
            spacing3=2, lmargin1=6, lmargin2=6,
        )
        text.tag_configure("warning_pad", background=_PACKAGE_WARNING_BG, spacing3=6)

        text.insert("end", f"{section_title}\n", "h1")

        if requirement:
            text.insert("end", "⚠ Installation supplémentaire nécessaire — sans elle, ça ne fonctionne pas\n", "warning_title")
            text.insert("end", requirement["note"] + "\n", "warning_body")
            for i, (label, url) in enumerate(requirement["links"]):
                link_tag = f"link_{i}"
                text.tag_configure(
                    link_tag, font=("TkDefaultFont", 9, "underline"),
                    foreground=_PACKAGE_WARNING_COLOR, background=_PACKAGE_WARNING_BG,
                    lmargin1=6, lmargin2=6,
                )
                text.tag_bind(link_tag, "<Button-1>", lambda _e, u=url: webbrowser.open(u))
                text.tag_bind(link_tag, "<Enter>", lambda _e: text.configure(cursor="hand2"))
                text.tag_bind(link_tag, "<Leave>", lambda _e: text.configure(cursor=""))
                is_last = i == len(requirement["links"]) - 1
                text.insert("end", f"→ {label}\n", (link_tag, "warning_pad") if is_last else (link_tag,))
            text.insert("end", "\n", "warning_pad")

        for label, explanation in entries:
            text.insert("end", f"{label}\n", "h2")
            text.insert("end", explanation + "\n", "body")
        text.configure(state="disabled")

        ttk.Button(self, text="Fermer", command=self.destroy).pack(pady=(0, 10))
        self.grab_set()


class SettingsTab(ttk.Frame):
    """Paramètres techniques de l'application, enregistrés dans
    config.json. "Enregistrer" les applique immédiatement aux opérations
    suivantes (entraînement, classification, automatisation...), sans avoir
    à redémarrer l'application. Les réglages bas niveau qui pourraient
    casser l'extraction ou la vectorisation si mal réglés restent dans le
    code plutôt que d'être exposés ici."""

    def __init__(self, parent):
        super().__init__(parent)
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill="both", expand=True)
        self.body = scrollable.body
        self.vars: dict[str, tk.Variable] = {}
        self._build()
        self._load_from_config()

    def _build(self) -> None:
        container = ttk.Frame(self.body)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        clustering = ttk.LabelFrame(container, text="Regroupement automatique (Entraînement)", padding=10)
        clustering.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_help_button(clustering, "clustering", "Regroupement automatique (Entraînement)")
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
        self._add_bool_field(
            clustering, 5, "cluster_use_keybert",
            "Noms de catégorie plus naturels via KeyBERT (repli TF-IDF si non installé)",
        )
        self._add_bool_field(
            clustering, 6, "cluster_detect_duplicates",
            "Détecter les documents en double pendant l'entraînement",
        )
        self._add_bool_field(
            clustering, 7, "train_rewrite_config_default",
            'Case "Réécrire la configuration" cochée par défaut (Entraînement)',
        )
        self._add_combo_field(
            clustering, 8, "cluster_algorithm", "Algorithme de regroupement :",
            ["kmeans", "minibatch_kmeans", "agglomerative", "hdbscan"],
        )
        self._add_int_field(
            clustering, 9, "cluster_large_corpus_threshold",
            "Bascule kmeans → minibatch_kmeans au-delà de ce nombre de documents :",
        )
        self._add_bool_field(
            clustering, 10, "cluster_use_svd",
            "Réduire la dimension des vecteurs TF-IDF avant regroupement (SVD/LSA)",
        )
        self._add_int_field(clustering, 11, "cluster_svd_components", "Composantes SVD :")
        self._add_bool_field(
            clustering, 12, "cluster_report_extra_metrics",
            "Afficher les métriques Davies-Bouldin/Calinski-Harabasz en plus de la silhouette",
        )
        self._add_str_field(
            clustering, 13, "cluster_extra_stopwords", "Mots à ignorer en plus (séparés par des virgules) :",
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
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(6, 0))

        vectorization = ttk.LabelFrame(container, text="Vectorisation", padding=10)
        vectorization.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self._add_help_button(vectorization, "vectorization", "Vectorisation")
        self._add_int_field(vectorization, 0, "tfidf_max_features", "Vocabulaire TF-IDF maximal :")
        self._add_int_field(vectorization, 1, "tfidf_ngram_max", "Taille maximale des groupes de mots (n-grammes) :")
        self._add_combo_field(
            vectorization, 2, "embedding_model_default", "Modèle d'embeddings par défaut :",
            [name for name, _description in EMBEDDING_MODEL_CATALOG],
        )
        self._add_bool_field(
            vectorization, 3, "tfidf_use_stemming",
            "Racinisation des mots (FR/EN) avant la vectorisation TF-IDF",
        )
        self._add_bool_field(
            vectorization, 4, "embedding_use_gpu",
            "Utiliser le GPU (CUDA) pour le moteur embeddings si disponible",
        )
        self._add_combo_field(
            vectorization, 5, "image_model_default", "Modèle CLIP par défaut (moteur image) :",
            [name for name, _description in IMAGE_MODEL_CATALOG],
        )
        self._add_str_field(
            vectorization, 6, "image_cluster_labels", "Libellés candidats (moteur image, virgules) :",
        )

        classification = ttk.LabelFrame(container, text="Classification", padding=10)
        classification.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_help_button(classification, "classification", "Classification")
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
        self._add_help_button(folders, "folders", "Dossiers de sortie")
        self._add_str_field(folders, 0, "models_root", "Dossier racine des modèles (storage/models/<nom>/) :")
        self._add_str_field(folders, 1, "default_output_dir", "Dossier de sortie par défaut (Classification) :")

        automation = ttk.LabelFrame(container, text="Automatisation", padding=10)
        automation.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_help_button(automation, "automation", "Automatisation")
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
        self._add_help_button(misc, "misc", "Divers")
        self._add_int_field(misc, 0, "model_discovery_max_depth", "Profondeur de recherche des modèles .pkl :")
        self._add_int_field(misc, 1, "model_history_keep", "Instantanés conservés par modèle (0 = désactivé) :")
        self._add_int_field(misc, 2, "window_width", "Largeur de fenêtre par défaut :")
        self._add_int_field(misc, 3, "window_height", "Hauteur de fenêtre par défaut :")
        self._add_int_field(misc, 4, "api_port", "Port du serveur API local (onglet API) :")

        extraction = ttk.LabelFrame(container, text="Extraction", padding=10)
        extraction.grid(row=3, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._add_help_button(extraction, "extraction", "Extraction")
        self._add_int_field(extraction, 0, "extraction_parallel_workers", "Fichiers extraits en parallèle :")

        ocr = ttk.LabelFrame(container, text="OCR (PDF scannés et images)", padding=10)
        ocr.grid(row=3, column=1, sticky="nsew", pady=(0, 10))
        self._add_help_button(ocr, "ocr", "OCR (PDF scannés et images)")
        self._add_bool_field(ocr, 0, "ocr_enabled", "Activer l'OCR (PDF scannés, images)")
        self._add_str_field(ocr, 1, "tesseract_cmd_path", "Chemin de l'exécutable Tesseract (si hors PATH) :")

        actions = ttk.Frame(self.body)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Enregistrer", command=self._save).pack(side="left")
        ttk.Button(actions, text="Réinitialiser aux valeurs par défaut", command=self._reset_defaults).pack(
            side="left", padx=6
        )

        self.status = tk.StringVar(value=f"Fichier de configuration : {os.path.abspath(DEFAULT_CONFIG_PATH)}")
        ttk.Label(self.body, textvariable=self.status, foreground="#555").pack(fill="x", pady=(8, 0))

    def _add_help_button(self, parent: ttk.LabelFrame, section_key: str, section_title: str) -> None:
        """Bouton "?" en haut à droite d'une section de paramètres, ouvrant
        une modale qui explique chaque réglage technique de cette section en
        détail (voir SETTINGS_HELP) — pour garder le formulaire lisible tout
        en gardant l'explication complète à portée de clic.

        Orange (icône ET contour), pour les sections listées dans
        `SETTINGS_PACKAGE_REQUIREMENTS` : au moins un réglage de cette
        section ne fonctionne pas du tout sans un paquet optionnel — un
        `tk.Button` classique plutôt que `ttk.Button` ici, seul moyen fiable
        (indépendant du thème ttk actif) d'obtenir un contour de couleur."""
        requirement = SETTINGS_PACKAGE_REQUIREMENTS.get(section_key)
        command = lambda: FieldHelpDialog(self, section_title, SETTINGS_HELP[section_key], requirement)
        if requirement:
            tk.Button(
                parent, text="?", width=2, command=command,
                foreground=_PACKAGE_WARNING_COLOR, activeforeground=_PACKAGE_WARNING_COLOR,
                highlightbackground=_PACKAGE_WARNING_COLOR, highlightcolor=_PACKAGE_WARNING_COLOR,
                highlightthickness=2, relief="solid", borderwidth=1, cursor="hand2",
            ).grid(row=0, column=2, rowspan=2, sticky="ne", padx=(10, 0))
        else:
            ttk.Button(parent, text="?", width=3, command=command).grid(
                row=0, column=2, rowspan=2, sticky="ne", padx=(10, 0)
            )

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

        # Classification reste toujours le premier onglet (à gauche) — il
        # regroupe désormais aussi la gestion des catégories détectées et de
        # leurs fichiers (ex-onglet "Transformer les données"), puisque les
        # deux s'appliquent au même modèle chargé. Le reste du flux de
        # travail (entraîner → automatiser) est regroupé à sa droite, dans
        # l'ordre où on s'en sert.
        self.classify_tab = ClassifyTab(notebook, on_model_improved=self._on_model_trained)
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
        self.classify_tab._load_model(model_path)
        # Sélectionne ce modèle comme "base à améliorer" dans l'onglet
        # Entraînement (même s'il vient d'être créé sans base_model_path, ou
        # amélioré depuis l'onglet Classification) : sans ça, ses catégories,
        # son aperçu 2D et son historique/rollback resteraient inaccessibles
        # tant qu'on ne le resélectionne pas à la main. `.set()` déclenche
        # `_on_base_model_path_changed` même si la valeur ne change pas
        # (rafraîchit alors quand même categories/aperçu/historique).
        self.train_tab.base_model_path.set(model_path)

    def _on_close(self) -> None:
        self.automation_tab.shutdown()
        self.api_tab.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
