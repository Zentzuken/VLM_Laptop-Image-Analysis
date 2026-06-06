import json

with open("dataset/database.json") as f:
    db = json.load(f)

cpu_count = 0
gpu_count = 0

for item in db:

    specs = item.get("specs", {})

    if specs.get("cpu"):
        cpu_count += 1

    if specs.get("gpu"):
        gpu_count += 1

print("TOTAL:", len(db))
print("CPU:", cpu_count)
print("GPU:", gpu_count)