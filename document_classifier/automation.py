"""Automatisation : surveille un dossier et applique un modèle à intervalle
régulier, sans intervention. Plusieurs automatisations indépendantes peuvent
tourner en même temps (chacune dans son propre thread), et leur configuration
est sauvegardée sur disque pour survivre à un redémarrage de l'application.
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .classify import (
    classify_documents,
    load_model_for_prediction,
    model_extensions,
    uncertain_category,
    unreadable_category,
)
from .config import get_config
from .extraction import extract_documents
from .utils import dispatch_file, write_json_atomic

UNIT_SECONDS = {"minutes": 60, "heures": 3600, "jours": 86400}
DEFAULT_AUTOMATIONS_PATH = "automations.json"


@dataclass
class AutomationConfig:
    name: str
    watch_dir: str
    model_path: str
    output_dir: str
    interval_value: int = 10
    interval_unit: str = "minutes"  # "minutes" | "heures" | "jours"
    move: bool = True
    threshold: float = 0.4
    recursive: bool = False
    enabled: bool = True
    # Si False, les fichiers classés "a_verifier" (confiance sous le seuil)
    # ne sont pas dispatchés — ils restent dans le dossier surveillé et
    # seront retentés au passage suivant plutôt que d'être exclus définitivement.
    include_uncertain: bool = True
    # Si False, les fichiers "non catégorisé" (texte illisible) ne sont pas
    # dispatchés — ils restent dans le dossier surveillé et sont retentés au
    # passage suivant.
    include_unreadable: bool = True

    def interval_seconds(self) -> float:
        return max(1, self.interval_value) * UNIT_SECONDS[self.interval_unit]


class AutomationJob:
    """Exécute une AutomationConfig dans un thread dédié, à intervalle régulier.

    Ne retraite jamais un fichier déjà vu (mémorisé dans `_seen`) : sans ça,
    en mode copie, le même fichier serait re-classé à chaque passage tant
    qu'il reste dans le dossier surveillé.
    """

    def __init__(self, config: AutomationConfig, on_event=print):
        self.config = config
        self.on_event = on_event
        self.bundle = None
        self.engine = None
        self.last_run: str | None = None
        self.last_run_count = 0
        self.last_error: str | None = None
        self._seen: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.bundle, self.engine = load_model_for_prediction(self.config.model_path)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.on_event(f"[{self.config.name}] Démarrée (toutes les {self.config.interval_value} {self.config.interval_unit}).")

    def stop(self) -> None:
        self._stop_event.set()
        self.on_event(f"[{self.config.name}] Arrêtée.")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception as exc:
                self.last_error = str(exc)
                self.on_event(f"[{self.config.name}] Erreur : {exc}")
            self._stop_event.wait(self.config.interval_seconds())

    def _run_once(self) -> None:
        documents = extract_documents(
            self.config.watch_dir, recursive=self.config.recursive, extensions=model_extensions(self.bundle)
        )
        new_documents = [d for d in documents if d.path not in self._seen]
        self.last_run = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not new_documents:
            self.last_run_count = 0
            return

        results = classify_documents(self.bundle, self.engine, new_documents, threshold=self.config.threshold)
        uncertain_name = uncertain_category()
        unreadable_name = unreadable_category()
        dispatched: dict[str, dict] = {}
        skipped_uncertain = 0
        skipped_unreadable = 0
        for path, info in results.items():
            category = info["category"]
            # Laissés en place, sans marquer comme "vus" : retentés au
            # prochain passage (utile si le modèle s'améliore entre-temps,
            # ou si le document redevient lisible après correction manuelle).
            if category == uncertain_name and not self.config.include_uncertain:
                skipped_uncertain += 1
                continue
            if category == unreadable_name and not self.config.include_unreadable:
                skipped_unreadable += 1
                continue
            dispatch_file(path, category, self.config.output_dir, move=self.config.move)
            self._seen.add(path)
            dispatched[path] = info

        if dispatched:
            manifest_path = self.config.output_dir.rstrip("/\\") + "_classification.json"
            write_json_atomic(dispatched, manifest_path)

        self.last_run_count = len(dispatched)
        message = f"[{self.config.name}] {len(dispatched)} nouveau(x) fichier(s) classé(s)."
        if skipped_uncertain:
            message += f" {skipped_uncertain} fichier(s) \"à vérifier\" laissé(s) de côté."
        if skipped_unreadable:
            message += f" {skipped_unreadable} fichier(s) \"non catégorisé\" laissé(s) de côté."
        self.on_event(message)


class AutomationManager:
    """Garde la liste des automatisations configurées, les persiste sur
    disque, et démarre automatiquement celles marquées comme actives."""

    def __init__(self, config_path: str | None = None, on_event=print):
        self.config_path = config_path if config_path is not None else get_config().automation_config_path
        self.on_event = on_event
        self.jobs: dict[str, AutomationJob] = {}

    def load(self) -> None:
        if not os.path.exists(self.config_path):
            return
        import json

        with open(self.config_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            config = AutomationConfig(**item)
            job = AutomationJob(config, on_event=self.on_event)
            self.jobs[config.name] = job
            if config.enabled:
                try:
                    job.start()
                except Exception as exc:
                    self.on_event(f"[{config.name}] Impossible de démarrer au lancement : {exc}")

    def save(self) -> None:
        data = [asdict(job.config) for job in self.jobs.values()]
        write_json_atomic(data, self.config_path)

    def add(self, config: AutomationConfig) -> AutomationJob:
        if config.name in self.jobs:
            raise ValueError(f"Une automatisation nommée {config.name!r} existe déjà.")
        job = AutomationJob(config, on_event=self.on_event)
        self.jobs[config.name] = job
        self.save()
        return job

    def remove(self, name: str) -> None:
        job = self.jobs.pop(name, None)
        if job:
            job.stop()
        self.save()

    def stop_all(self) -> None:
        for job in self.jobs.values():
            job.stop()
