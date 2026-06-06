import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPProcessor,
    CLIPModel
)

# CONFIG
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 20 #5 
LR = 1e-5


# LOAD DATA
with open("dataset/database.json", "r") as f:
    database = json.load(f)


# DATASET
class LaptopDataset(Dataset):

    def __init__(self, data):
        self.samples = []

        for item in data:

            title = item.get("title", "")

            specs = item.get("specs", {})

            gpu = specs.get("gpu", "")
            cpu = specs.get("cpu", "")
            category = specs.get("category", "")

            text = (
                f"{title} "
                f"{gpu} "
                f"{cpu} "
                f"{category} laptop"
            )

            for img_path in item["images"]:

                if os.path.exists(img_path):

                    self.samples.append({
                        "image": img_path,
                        "text": text
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(
            sample["image"]
        ).convert("RGB")

        return image, sample["text"]

dataset = LaptopDataset(database)

print("TRAINING SAMPLES:", len(dataset))


# MODEL
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# DATALOADER
def collate_fn(batch):

    images = [x[0] for x in batch]
    texts = [x[1] for x in batch]

    return processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# TRAIN
model.train()

for epoch in range(EPOCHS):

    print(f"\nEPOCH {epoch+1}")

    total_loss = 0

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        logits_per_image = outputs.logits_per_image

        labels = torch.arange(
            len(logits_per_image)
        ).to(DEVICE)

        loss_i = torch.nn.functional.cross_entropy(
            logits_per_image,
            labels
        )

        loss_t = torch.nn.functional.cross_entropy(
            outputs.logits_per_text,
            labels
        )

        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("LOSS:", total_loss)


# SAVE
model.save_pretrained(
    "fine_tuned_clip"
)

processor.save_pretrained(
    "fine_tuned_clip"
)

print("\nMODEL SAVED")