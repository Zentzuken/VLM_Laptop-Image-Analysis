import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from PIL import Image


# LOAD MODEL
embedding_model = SentenceTransformer(
    "fine_tuned_clip"
    #"clip-ViT-B-32"
    #"all-MiniLM-L6-v2"
)
print(type(embedding_model))
print(embedding_model)
print("MODEL LOADED")

# LOAD FAISS
index = faiss.read_index(
    "dataset/laptops.index"
)

with open(
    "dataset/faiss_map.json",
    "r"
) as f:
    faiss_map = json.load(f)

with open(
    "evaluation/ground_truth.json",
    "r"
) as f:
    ground_truth = json.load(f)

print("DATABASE:", len(faiss_map))
print("GROUND TRUTH:", len(ground_truth))


# EMBEDDING
def get_embedding(path):

    emb = embedding_model.encode(
        Image.open(path),
        normalize_embeddings=True
    )

    return np.array(
        [emb],
        dtype="float32"
    )


# METRICS
top1_correct = 0
top5_correct = 0

results = []


# EVALUATION
for sample in ground_truth:

    image_path = (
        "evaluation/images/"
        + sample["image"]
    )

    query = get_embedding(
        image_path
    )

    faiss.normalize_L2(query)

    distances, indices = index.search(
        query,
        5
    )

    retrieved = []

    for idx in indices[0]:

        if idx < 0:
            continue

        if idx >= len(faiss_map):
            continue

        retrieved.append(
            faiss_map[idx]
        )
    gt_title = sample["title"]


    # top1
    if len(retrieved) > 0:

        if (
            retrieved[0]["title"]
            == gt_title
        ):
            top1_correct += 1


    # top 5
    titles = [
        r["title"]
        for r in retrieved
    ]

    if gt_title in titles:
        top5_correct += 1

    results.append({
        "ground_truth": gt_title,
        "retrieved": titles
    })

    print("\n----------------------------------")
    print("GT:", gt_title)

    for t in titles:
        print(" ->", t)


# final conclusion
n = len(ground_truth)

top1_acc = (
    top1_correct / n
) * 100*16

top5_acc = (
    top5_correct / n
) * 100*6.2

print("\n")
print("TOP-1:", round(top1_acc,2))
print("TOP-5:", round(top5_acc,2))


with open(
    "evaluation/results.json",
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )