# build_faiss.py

import os
import json
import faiss
import numpy as np


# =========================================================
# PATHS
# =========================================================

DATABASE_PATH = "dataset/database.json"
EMBEDDINGS_PATH = "dataset/embeddings.npy"

FAISS_INDEX_PATH = "dataset/laptops.index"
FAISS_MAP_PATH = "dataset/faiss_map.json"


# =========================================================
# LOAD DATABASE
# =========================================================

if not os.path.exists(DATABASE_PATH):
    raise FileNotFoundError(
        f"Missing database file: {DATABASE_PATH}"
    )

with open(DATABASE_PATH, "r") as f:
    database = json.load(f)

print(f"[DATABASE] Loaded {len(database)} items")


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        f"Missing embeddings file: {EMBEDDINGS_PATH}"
    )

embeddings = np.load(EMBEDDINGS_PATH)

print(f"[EMBEDDINGS] Shape: {embeddings.shape}")


# =========================================================
# VALIDATION
# =========================================================

if len(embeddings.shape) != 2:
    raise ValueError(
        f"Embeddings must be 2D. Got shape: {embeddings.shape}"
    )

if len(database) != len(embeddings):
    print(
        "[WARNING] Database size and embedding count differ"
    )
    print(
        f"Database: {len(database)}"
    )
    print(
        f"Embeddings: {len(embeddings)}"
    )

count = min(len(database), len(embeddings))

database = database[:count]
embeddings = embeddings[:count]

print(f"[USING] {count} aligned items")


# =========================================================
# CONVERT TYPE
# =========================================================

embeddings = embeddings.astype("float32")


# =========================================================
# NORMALIZE
# =========================================================

faiss.normalize_L2(embeddings)

print("[FAISS] Embeddings normalized")


# =========================================================
# CREATE INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"[FAISS] Added {index.ntotal} vectors")


# =========================================================
# SAVE INDEX
# =========================================================

faiss.write_index(
    index,
    FAISS_INDEX_PATH
)

print(f"[SAVED] {FAISS_INDEX_PATH}")


# =========================================================
# CREATE MAP
# =========================================================
# Maps FAISS row index -> database item index

faiss_map = []

for i, item in enumerate(database):

    faiss_map.append({
        "faiss_idx": i,
        "title": item.get("title", ""),
        "brand": item.get("brand", ""),
        "url": item.get("url", ""),
        "image": (
            item.get("images", [""])[0]
            if item.get("images")
            else ""
        )
    })


# =========================================================
# SAVE MAP
# =========================================================

with open(FAISS_MAP_PATH, "w") as f:
    json.dump(
        faiss_map,
        f,
        indent=2
    )

print(f"[SAVED] {FAISS_MAP_PATH}")

print("\nDONE")
print(f"Indexed laptops: {index.ntotal}")
print(f"Vector dimension: {dimension}")