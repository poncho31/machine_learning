import fitz
import os
import numpy as np
import json
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 1. Extraction ──────────────────────────────────────────────
def extract_text(path: str) -> str:
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

pdf_dir = "./pdfs"
output_dir = "./classified"

files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
texts = [extract_text(os.path.join(pdf_dir, f)) for f in files]

# ── 2. Embeddings ──────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = normalize(model.encode(texts, show_progress_bar=True))

# ── 3. Clustering ──────────────────────────────────────────────
N_CLUSTERS = 5
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
labels = km.fit_predict(embeddings)

# ── 4. Nommage automatique des clusters via TF-IDF ─────────────
def name_clusters(texts, labels, n_clusters, top_n=3):
    """
    Pour chaque cluster, concatène tous ses textes,
    puis extrait les top_n mots TF-IDF les plus représentatifs.
    Le nom = ces mots joints par underscore.
    """
    cluster_texts = [""] * n_clusters
    for text, label in zip(texts, labels):
        cluster_texts[label] += " " + text

    # TF-IDF sur les "super-documents" (1 par cluster)
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",       # remplacer par une liste FR si corpus français
        ngram_range=(1, 2),         # unigrammes + bigrammes
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    terms = vectorizer.get_feature_names_out()

    cluster_names = {}
    for cluster_id in range(n_clusters):
        # Scores TF-IDF pour ce cluster
        scores = tfidf_matrix[cluster_id].toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_words = [terms[i] for i in top_indices]
        cluster_names[cluster_id] = "_".join(top_words)

    return cluster_names

cluster_names = name_clusters(texts, labels, N_CLUSTERS)

print("\n── Catégories détectées ──")
for cid, name in cluster_names.items():
    print(f"  Cluster {cid} → {name}")

# ── 5. Classement des fichiers ─────────────────────────────────
print("\n── Classification ──")
for filename, label in zip(files, labels):
    category = cluster_names[label]
    dest = os.path.join(output_dir, category)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(os.path.join(pdf_dir, filename), dest)
    print(f"  {filename} → {category}/")

# ── 6. Export du modèle (pour réutilisation) ───────────────────
import pickle

model_data = {
    "cluster_names": cluster_names,
    "km_model": km,
    "labels": labels.tolist(),
    "files": files,
}
with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Export lisible
with open("classification.json", "w") as f:
    json.dump({
        "categories": cluster_names,
        "files": {f: cluster_names[l] for f, l in zip(files, labels)}
    }, f, indent=2, ensure_ascii=False)

print("\n✓ model.pkl + classification.json sauvegardés")