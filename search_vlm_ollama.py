import json
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer

from transformers import CLIPProcessor, CLIPModel

from PIL import Image
import torch

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# device setup BROKEN ----------------
#if torch.backends.mps.is_available():
#    DEVICE = "mps"
#    DEVICE = "cpu"

#elif torch.cuda.is_available():
#    DEVICE = "cuda"

#else:
#    DEVICE = "cpu"


# device setup OVERWRITE
DEVICE = "cpu"

# load clip
#embedding_model = CLIPModel.from_pretrained(
#    "openai/clip-vit-base-patch32"
#).to(DEVICE)


# load db
#with open("dataset/faiss_map.json", "r") as f:
#    database = json.load(f)

# load FAISS
index = faiss.read_index(
    "dataset/laptops.index"
)

with open("dataset/faiss_map.json", "r") as f:
    faiss_map = json.load(f)

#item = faiss_map[idx]

print("FAISS vectors:", index.ntotal)

with open("dataset/faiss_map.json", "r") as f:
    faiss_map = json.load(f)
print("FAISS map:", len(faiss_map))

with open("dataset/database.json", "r") as f:
    database = json.load(f)
print("Database:", len(database))


embedding_model = SentenceTransformer(
    #"clip-ViT-B-32"
    "fine_tuned_clip"
    #"all-MiniLM-L6-v2"
)

print("EMBEDDING MODEL LOADED")


# embedding
def get_embedding(image_path):

    emb = embedding_model.encode(
        Image.open(image_path),
        normalize_embeddings=True
    )

    return emb

# multiple images
def get_model_embedding(image_paths):

    embeddings = []

    for path in image_paths:

        try:
            emb = get_embedding(path)
            embeddings.append(emb)

        except Exception as e:
            print(e)

    if not embeddings:
        return None

    embeddings = np.array(embeddings)

    final_embedding = embeddings.mean(axis=0)

    final_embedding = (
        final_embedding /
        np.linalg.norm(final_embedding)
    )

    return final_embedding.astype("float32")

print("EMBEDDING CREATED")



# query and search image
query_image = "query.jpg"

query_embedding = get_embedding(query_image)

query_embedding = np.array(
    [query_embedding],
    dtype="float32"
)

faiss.normalize_L2(query_embedding)

distances, indices = index.search(
    query_embedding,
    5
)


# retrieval
retrieved = []

for idx in indices[0]:
    # invalid index protection
    if idx < 0:
        continue

    if idx >= len(faiss_map):
        continue

    item = faiss_map[idx]
    retrieved.append(item)


#  context
context = ""

for item in retrieved:

    context += f"""
    Laptop: {item.get('title')}
    Brand: {item.get('brand')}
    Specs:
    {json.dumps(item.get('specs', {}), indent=2)}
    URL:
    {item.get('url')}
    """

print("\nRESULTS")

for score, idx in zip(distances[0], indices[0]):

    print(
        score,
        idx,
        faiss_map[idx]["title"]
    )

#print("TESTING OLLAMA CONNECTION")
#models = ollama.list()
#print(models)


print("ASKING MODEL")
# =====================================================
# QWEN VLM
response = ollama.chat(
    model="qwen2.5vl:3b",
    messages=[
        {
            "role": "user",
            "content": f"""
You are a laptop recommendation AI.

Retrieved similar laptops:

{context}

Analyze the uploaded laptop image, provide every detail available of the uploaded laptop first over everything.
Add and use information from online sources whenever capable.
Compare queried laptop to retrieved laptops, focusing on performance, price range, and longevity.
""",
            "images": [query_image]
        }
    ]
)

print("\n==============================")
print(response["message"]["content"])
print("\n")