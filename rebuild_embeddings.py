import json
import faiss
import numpy as np

from PIL import Image

import torch

from transformers import (
    CLIPModel,
    CLIPProcessor
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# load dataset
with open("dataset/database.json", "r") as f:
    database = json.load(f)

# load trained model
model = CLIPModel.from_pretrained(
    "fine_tuned_clip"
    #CLIP ViT-B/32
    #"all-MiniLM-L6-v2"

).to(DEVICE)

processor = CLIPProcessor.from_pretrained(
    "fine_tuned_clip"
)

print("MODEL LOADED")


def get_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        emb = model.get_image_features(
            **inputs
        )

    emb = emb / emb.norm(
        dim=-1,
        keepdim=True
    )

    return emb.cpu().numpy()[0]


embeddings = []
faiss_map = []

for item in database:

    image_embeddings = []

    for img_path in item["images"]:

        try:

            emb = get_embedding(img_path)

            image_embeddings.append(
                emb
            )

        except Exception as e:

            print(e)

    if not image_embeddings:
        continue

    final_embedding = np.mean(
        image_embeddings,
        axis=0
    )

    final_embedding = (
        final_embedding /
        np.linalg.norm(final_embedding)
    )

    embeddings.append(
        final_embedding.astype("float32")
    )

    faiss_map.append(item)

print("EMBEDDINGS:", len(embeddings))

embeddings = np.array(
    embeddings,
    dtype="float32"
)

# cosine similarity
faiss.normalize_L2(
    embeddings
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "dataset/laptops.index"
)

with open(
    "dataset/faiss_map.json",
    "w"
) as f:

    json.dump(
        faiss_map,
        f,
        indent=2
    )

print("FAISS REBUILT")