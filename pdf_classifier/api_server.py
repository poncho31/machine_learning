"""Serveur API HTTP local : permet à un autre programme de piloter
l'application (entraînement, classification, transformation des catégories,
historique) sans réimplémenter sa logique — il suffit d'appeler ces routes.

N'écoute que sur 127.0.0.1 (jamais exposé au réseau). Toutes les routes sauf
"/" et "/health" exigent une clé, régénérée à chaque démarrage du serveur et
affichée dans l'onglet API, envoyée en en-tête `Authorization: Bearer <clé>`
(ou `X-API-Key: <clé>`) — sans quoi n'importe quel programme tournant sur la
machine pourrait piloter l'application à votre place.

Chaque requête est traitée dans son propre thread (`ThreadingHTTPServer`),
mais de façon synchrone : la réponse n'arrive qu'une fois l'opération
terminée (un entraînement peut donc prendre du temps à répondre).
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import model_store
from .classify import classify as classify_fn
from .discover import build_model as build_model_fn
from .discover import improve_model as improve_model_fn
from .extraction import extract_documents_from_paths
from .rename import delete_category, rename_categories

ROUTES: list[tuple[str, str, str]] = [
    ("GET", "/", "Liste toutes les routes disponibles (ce document), sans clé requise."),
    ("GET", "/health", "Vérifie que le serveur répond, sans clé requise."),
    ("GET", "/models", "Liste les modèles trouvés : nom, chemin, taille en octets."),
    ("GET", "/models/<name>", "Détails d'un modèle : catégories, nombre de documents, date de création."),
    ("GET", "/models/<name>/snapshots", "Instantanés d'historique disponibles pour ce modèle (les plus récents en premier)."),
    ("POST", "/models/<name>/restore", 'Restaure un instantané. Corps : {"timestamp": "..."}'),
    (
        "POST", "/train",
        'Entraîne ou améliore un modèle (aucun tri manuel requis). Corps : '
        '{"input_dir": "...", "model_name": "...", "engine": "tfidf"|"embeddings", '
        '"embedding_model": "..." (optionnel), "base_model_name": "..." (optionnel, améliore ce modèle existant)}',
    ),
    (
        "POST", "/classify",
        'Classe les documents d\'un dossier avec un modèle. Corps : '
        '{"model_name": "...", "input_dir": "...", "output_dir": "...", '
        '"threshold": 0.4 (optionnel), "move": false (optionnel), "recursive": false (optionnel)}',
    ),
    (
        "POST", "/improve",
        'Améliore un modèle avec des catégories déjà confirmées (prime sur toute prédiction). Corps : '
        '{"model_name": "...", "files": [{"path": "...", "category": "..."}, ...]}',
    ),
    (
        "POST", "/rename",
        'Renomme une ou plusieurs catégories. Corps : '
        '{"model_name": "...", "renames": {"ancien_nom": "nouveau_nom", ...}}',
    ),
    (
        "POST", "/delete-category",
        'Supprime une catégorie (fusionnée dans "autre", rien n\'est perdu). Corps : '
        '{"model_name": "...", "category": "...", "other_name": "autre" (optionnel)}',
    ),
]


def _resolve_model_path(name: str) -> str:
    path = model_store.model_path_for_name(name)
    if os.path.exists(path):
        return path
    if os.path.exists(name):  # un chemin complet a été fourni directement
        return name
    raise FileNotFoundError(f"Modèle introuvable : {name!r}")


class _Handler(BaseHTTPRequestHandler):
    server_version = "PDFClassifierAPI/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (signature imposée par BaseHTTPRequestHandler)
        on_event = getattr(self.server, "on_event", None)
        if on_event:
            on_event(f"{self.address_string()} — {format % args}")

    # ── Aides ──
    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _check_auth(self) -> bool:
        token = getattr(self.server, "token", None)
        if not token:
            return True
        auth_header = self.headers.get("Authorization", "")
        provided = auth_header[7:] if auth_header.startswith("Bearer ") else self.headers.get("X-API-Key", "")
        return secrets.compare_digest(provided, token)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        if method == "GET" and not parts:
            self._send_json(200, {"routes": [{"method": m, "path": p, "description": d} for m, p, d in ROUTES]})
            return
        if method == "GET" and parts == ["health"]:
            self._send_json(200, {"status": "ok"})
            return

        if not self._check_auth():
            self._send_json(
                401,
                {"error": "Clé API manquante ou invalide (en-tête 'Authorization: Bearer <clé>' ou 'X-API-Key')."},
            )
            return

        try:
            self._route(method, parts)
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _route(self, method: str, parts: list[str]) -> None:
        if method == "GET" and parts == ["models"]:
            self._handle_list_models()
        elif method == "GET" and len(parts) == 2 and parts[0] == "models":
            self._handle_model_detail(parts[1])
        elif method == "GET" and len(parts) == 3 and parts[0] == "models" and parts[2] == "snapshots":
            self._handle_snapshots(parts[1])
        elif method == "POST" and len(parts) == 3 and parts[0] == "models" and parts[2] == "restore":
            self._handle_restore(parts[1])
        elif method == "POST" and parts == ["train"]:
            self._handle_train()
        elif method == "POST" and parts == ["classify"]:
            self._handle_classify()
        elif method == "POST" and parts == ["improve"]:
            self._handle_improve()
        elif method == "POST" and parts == ["rename"]:
            self._handle_rename()
        elif method == "POST" and parts == ["delete-category"]:
            self._handle_delete_category()
        else:
            self._send_json(404, {"error": f"Route inconnue : {method} /{'/'.join(parts)}"})

    # ── Modèles ──
    def _handle_list_models(self) -> None:
        models = model_store.discover_models(".")
        self._send_json(
            200,
            {
                "models": [
                    {"name": os.path.splitext(os.path.basename(path))[0], "path": path, "size_bytes": size}
                    for path, size in models
                ]
            },
        )

    def _handle_model_detail(self, name: str) -> None:
        path = _resolve_model_path(name)
        bundle = model_store.load_bundle(path)
        if bundle.get("mode") == "unsupervised":
            categories = list(bundle.get("cluster_names", {}).values())
        else:
            categories = list(bundle.get("label_names", []))
        self._send_json(
            200,
            {
                "name": name,
                "path": path,
                "mode": bundle.get("mode"),
                "categories": categories,
                "n_documents_trained": bundle.get("n_documents_trained"),
                "created_at": bundle.get("created_at"),
            },
        )

    def _handle_snapshots(self, name: str) -> None:
        path = _resolve_model_path(name)
        self._send_json(200, {"snapshots": model_store.list_snapshots(path)})

    def _handle_restore(self, name: str) -> None:
        path = _resolve_model_path(name)
        body = self._read_json_body()
        timestamp = body.get("timestamp")
        if not timestamp:
            raise ValueError("Le champ 'timestamp' est requis.")
        model_store.restore_snapshot(path, timestamp)
        self._send_json(200, {"status": "restored", "timestamp": timestamp})

    # ── Entraînement / classification ──
    def _handle_train(self) -> None:
        body = self._read_json_body()
        input_dir = body.get("input_dir")
        model_name = body.get("model_name")
        if not input_dir or not model_name:
            raise ValueError("Les champs 'input_dir' et 'model_name' sont requis.")
        model_path = model_store.model_path_for_name(model_name)
        base_model_path = model_store.model_path_for_name(body["base_model_name"]) if body.get("base_model_name") else None

        log: list[str] = []
        bundle = build_model_fn(
            input_dir=input_dir,
            model_path=model_path,
            engine_name=body.get("engine", "tfidf"),
            embedding_model=body.get("embedding_model"),
            base_model_path=base_model_path,
            progress=log.append,
        )
        self._send_json(
            200,
            {
                "model_name": model_name,
                "model_path": model_path,
                "categories": list(bundle["cluster_names"].values()),
                "n_documents_trained": bundle.get("n_documents_trained"),
                "log": log,
            },
        )

    def _handle_classify(self) -> None:
        body = self._read_json_body()
        model_name = body.get("model_name")
        input_dir = body.get("input_dir")
        output_dir = body.get("output_dir")
        if not model_name or not input_dir or not output_dir:
            raise ValueError("Les champs 'model_name', 'input_dir' et 'output_dir' sont requis.")
        model_path = _resolve_model_path(model_name)

        log: list[str] = []
        results = classify_fn(
            input_dir=input_dir,
            model_path=model_path,
            output_dir=output_dir,
            threshold=body.get("threshold"),
            recursive=bool(body.get("recursive", False)),
            move=bool(body.get("move", False)),
            progress=log.append,
        )
        self._send_json(200, {"results": results, "log": log})

    def _handle_improve(self) -> None:
        body = self._read_json_body()
        model_name = body.get("model_name")
        files = body.get("files")
        if not model_name or not files:
            raise ValueError("Les champs 'model_name' et 'files' sont requis.")
        model_path = _resolve_model_path(model_name)

        paths = [entry["path"] for entry in files]
        documents = extract_documents_from_paths(paths)
        confirmed_labels = {entry["path"]: entry["category"] for entry in files}

        log: list[str] = []
        bundle = improve_model_fn(model_path, documents, confirmed_labels, progress=log.append)
        self._send_json(200, {"categories": list(bundle["cluster_names"].values()), "log": log})

    # ── Transformer les données ──
    def _handle_rename(self) -> None:
        body = self._read_json_body()
        model_name = body.get("model_name")
        renames = body.get("renames")
        if not model_name or not renames:
            raise ValueError("Les champs 'model_name' et 'renames' sont requis.")
        model_path = _resolve_model_path(model_name)
        bundle = rename_categories(model_path, renames)
        self._send_json(200, {"categories": list(bundle["cluster_names"].values())})

    def _handle_delete_category(self) -> None:
        body = self._read_json_body()
        model_name = body.get("model_name")
        category = body.get("category")
        if not model_name or not category:
            raise ValueError("Les champs 'model_name' et 'category' sont requis.")
        model_path = _resolve_model_path(model_name)
        bundle = delete_category(model_path, category, other_name=body.get("other_name"))
        self._send_json(200, {"categories": list(bundle["cluster_names"].values())})


class ApiServer:
    """Cycle de vie du serveur API : `start()`/`stop()`, appelables depuis
    l'onglet API. Une nouvelle clé est générée à chaque `start()`."""

    def __init__(self, port: int, on_event=print):
        self.port = port
        self.on_event = on_event
        self.token: str = ""
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.token = secrets.token_urlsafe(24)
        httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        httpd.token = self.token  # type: ignore[attr-defined]
        httpd.on_event = self.on_event  # type: ignore[attr-defined]
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        self.on_event(f"Serveur API démarré sur http://127.0.0.1:{self.port}/")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._httpd = None
        self._thread = None
        self.on_event("Serveur API arrêté.")
