import os
import re
import json
import time
import requests
import numpy as np
#TODO import faiss
import hashlib
import shutil

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

import torch
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# CONFIG
# =========================================================


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}

OUTPUT_DIR = "dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

# GROUND TRUTH DATASET
GROUND_TRUTH_DIR = "evaluation"

GROUND_TRUTH_IMAGES = os.path.join(
    GROUND_TRUTH_DIR,
    "images"
)

os.makedirs(
    GROUND_TRUTH_IMAGES,
    exist_ok=True
)

os.makedirs(IMAGE_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

MIN_IMAGE_SIZE = 500
MAX_IMAGES_PER_PRODUCT = 7
MAX_DIM = 768


os.makedirs(
    GROUND_TRUTH_IMAGES,
    exist_ok=True
)

session = requests.Session()
session.headers.update(HEADERS)

# Load CLIP
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)



# Fetch url page
def fetch_page(url, retries=3):

    for attempt in range(retries):

        try:

            print(f"[FETCH] {url}")

            response = session.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            print(f"[FETCH OK] {url}")
            print(response.headers.get("Content-Type"))
            print(response.text[:500])

            return response.text

        except Exception as e:

            print(f"[FETCH ERROR] Attempt {attempt+1}")

            print(e)

            time.sleep(2)

    return None

# extract page JSON-LD
def extract_json_ld(soup):
    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    data_blocks = []

    for script in scripts:
        try:
            content = script.string

            if not content:
                continue

            data = json.loads(content)

            if isinstance(data, list):
                data_blocks.extend(data)
            else:
                data_blocks.append(data)

        except:
            continue

    return data_blocks


def flatten_jsonld(data_blocks):
    flat = []

    for item in data_blocks:
        if isinstance(item, dict) and "@graph" in item:
            flat.extend(item["@graph"])
        else:
            flat.append(item)

    return flat


def find_product_jsonld(data_blocks):
    for item in data_blocks:

        if not isinstance(item, dict):
            continue

        item_type = item.get("@type")

        if item_type in ["Product", "ProductModel"]:
            return item

    return None


def parse_jsonld(product_json):

    title = product_json.get("name", "")

    brand = "UNKNOWN"

    if isinstance(product_json.get("brand"), dict):
        brand = product_json["brand"].get(
            "name",
            "UNKNOWN"
        )

    images = product_json.get("image", [])

    if isinstance(images, str):
        images = [images]

    desc = product_json.get("description", "")

    return {
        "title": title,
        "brand": brand,
        "images": images,
        "description": desc
    }


# Image extract from all collected URL
def extract_all_images(soup, base_url):

    images = set()

    for img in soup.find_all("img"):

        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-image-url")
        )

        if not src:
            continue

        full_url = urljoin(base_url, src)

        lower = full_url.lower()

        # ---------------------------------------------
        # EXTENSION FILTER
        # ---------------------------------------------
        VALID_TERMS = [
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]

        if not any(term in lower for term in VALID_TERMS):
            continue

        # ---------------------------------------------
        # BAD IMAGE FILTER
        # ---------------------------------------------
        BAD_TERMS = [
            "logo",
            "icon",
            "banner",
            "ad",
            "promo",
            "thumbnail",
            "avatar",
            "payment",
            "tracking",
            "social"
        ]

        if any(term in lower for term in BAD_TERMS):
            continue

        images.add(full_url)

    return list(images)



def extract_notebookcheck_images(soup):

    images = []

    for img in soup.find_all("img"):

        src = img.get("src")

        if not src:
            continue

        src = src.lower()

        # =====================================
        # ONLY REAL LAPTOP IMAGES
        # =====================================
        if "/fileadmin/" not in src:
            continue

        if not any(
            ext in src
            for ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]
        ):
            continue

        # =====================================
        # FILTER BAD IMAGES
        # =====================================
        BAD = [
            "logo",
            "rating",
            "author",
            "youtube",
            "banner",
            "teaser",
            "icon"
        ]

        if any(b in src for b in BAD):
            continue

        images.append(src)

    return list(set(images))


#  fALLBACK PARSER
def parse_html_fallback(soup, url):

    title = (
        soup.title.text.strip()
        if soup.title
        else ""
    )

    text = soup.get_text(" ").lower()

    images = extract_all_images(soup, url)

    return {
        "title": title,
        "raw_text": text,
        "images": images
    }


# PARSE SPEC
def parse_specs(text, title):

    combined = (title + " " + text).lower()

    brands = [
        "asus",
        "dell",
        "lenovo",
        "hp",
        "acer",
        "msi",
        "apple",
        "razer"
    ]

    brand = next(
        (
            b.upper()
            for b in brands
            if b in combined
        ),
        "UNKNOWN"
    )

    gpu_match = re.search(
        r"(rtx\s?\d{3,4}|gtx\s?\d{3,4}|rx\s?\d{4}[a-z]?|arc\s?[a-z0-9]+)",
        combined,
        re.I
    )

    gpu = (
        gpu_match.group(0).upper()
        if gpu_match
        else None
    )

    cpu_patterns = [

        r"core ultra\s?[3579]\s?\d+[a-z]*",

        r"ryzen ai\s?[3579]\s?[a-z]*\s?\d+",

        r"ryzen\s?[3579]\s?\d{4}[a-z]*",

        r"i[3579]-\d{4,5}[a-z]*",

        r"snapdragon\s*x\s*elite",

        r"snapdragon\s*x\s*plus"
    ]

    cpu = None

    for pattern in cpu_patterns:

        match = re.search(
            pattern,
            combined,
            re.I
        )

        if match:
            cpu = match.group(0)
            break

    ram_match = re.search(
        r"(\d{1,2})\s?gb\s?ram",
        combined
    )

    ram = (
        ram_match.group(1) + "GB"
        if ram_match
        else None
    )

    year_match = re.search(r"(20[1-3][0-9])", combined)

    return {
        "brand": brand,
        "gpu": gpu,
        "cpu": cpu,
        "ram": ram
    }


# Category
def categorize(parsed):

    gpu_class = "integrated"

    if parsed["gpu"]:

        if any(
            x in parsed["gpu"]
            for x in ["4060", "4070", "4080"]
        ):
            gpu_class = "high"

        elif any(
            x in parsed["gpu"]
            for x in ["3050", "3060"]
        ):
            gpu_class = "mid"

    category = "business"

    if gpu_class == "high":
        category = "gaming"

    elif parsed["brand"] == "APPLE":
        category = "ultrabook"

    return {
        "gpu_class": gpu_class,
        "category": category
    }


# Clean IMAGE (not the actual image, but deletong images that don't match)
def validate_image(img):

    width, height = img.size

    # ---------------------------------------------
    # MINIMUM SIZE
    # ---------------------------------------------
    if width < MIN_IMAGE_SIZE:
        return False

    if height < MIN_IMAGE_SIZE:
        return False

    # ---------------------------------------------
    # ASPECT RATIO FILTER
    # ---------------------------------------------
    ratio = width / height

    if ratio > 4:
        return False

    if ratio < 0.25:
        return False

    return True


# IMAGE DOWNLOADER
MAX_DIM = 1024


def score_image(url):

    lower = url.lower()

    score = 0

    # =====================================================
    # GOOD TERMS
    # =====================================================
    GOOD_TERMS = [
        "feature",
        "overview",
        "front",
        "side",
        "back",
        "open",
        "closed",
        "hero",
        "gallery",
        "laptop",
        "notebook",
        "device",
        "chassis",
        "lid",
        "top",
        "bottom",
        "angle"
    ]

    for term in GOOD_TERMS:
        if term in lower:
            score += 5

    # =====================================================
    # BAD TERMS
    # =====================================================
    BAD_TERMS = [
        "stress",
        "benchmark",
        "cpuz",
        "gpuz",
        "thermal",
        "heat",
        "graph",
        "fps",
        "chart",
        "latency",
        "temperature",
        "keyboard",
        "ports",
        "screen",
        "colorchecker",
        "calman",
        "logo",
        "youtube",
        "speaker",
        "waveform",
        "ssd",
        "battery",
        "wifi",
        "noise",
        "power",
        "consumption",
        "cinebench",
        "3dmark",
        "flir",
        "dpc",
        "meter",
        "idle",
        "load",
        "performance",
        "analysis",
        "test",
        "gpu",
        "cpu"
    ]

    for term in BAD_TERMS:
        if term in lower:
            score -= 10

    # =====================================================
    # PREFER REAL PHOTO FORMATS
    # =====================================================
    if ".jpg" in lower or ".jpeg" in lower:
        score += 3

    if ".png" in lower:
        score -= 1

    # =====================================================
    # REJECT BAD FILE TYPES
    # =====================================================
    BAD_EXTENSIONS = [
        ".svg",
        ".gif"
    ]

    if any(ext in lower for ext in BAD_EXTENSIONS):
        score -= 100

    return score


# =========================================================
# COLLECT + RANK IMAGES
# =========================================================
def collect_all_images(soup, base_url, jsonld_images=None):

    all_images = set()

    # =====================================================
    # JSON-LD IMAGES
    # =====================================================
    if jsonld_images:

        if isinstance(jsonld_images, str):
            jsonld_images = [jsonld_images]

        for img in jsonld_images:

            if img.startswith("http"):
                all_images.add(img)

    # =====================================================
    # NOTEBOOKCHECK GALLERY IMAGES
    # =====================================================
    gallery = soup.select("div.csc-textpic-imagewrap img")

    for img in gallery:

        src = (
            img.get("src")
            or img.get("data-src")
        )

        if not src:
            continue

        src = urljoin(base_url, src)

        if "/fileadmin/" not in src:
            continue

        all_images.add(src)

    # =====================================================
    # META IMAGES
    # =====================================================
    meta_tags = soup.find_all("meta")

    for tag in meta_tags:

        content = tag.get("content")

        if not content:
            continue

        prop = (
            tag.get("property", "") +
            tag.get("name", "")
        ).lower()

        if "image" not in prop:
            continue

        if not content.startswith("http"):
            continue

        all_images.add(content)

    # =====================================================
    # NORMAL IMG TAGS
    # =====================================================
    html_images = extract_all_images(
        soup,
        base_url
    )

    for img in html_images:
        all_images.add(img)

    # =====================================================
    # FILTER INVALID FILE TYPES
    # =====================================================
    filtered = []

    VALID_EXTENSIONS = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]

    for img in all_images:

        lower = img.lower()

        if not any(ext in lower for ext in VALID_EXTENSIONS):
            continue

        if ".svg" in lower:
            continue

        if "logo" in lower:
            continue

        filtered.append(img)

    # =====================================================
    # RANK IMAGES
    # =====================================================
    ranked = sorted(
        filtered,
        key=score_image,
        reverse=True
    )

    print(f"[IMAGE CANDIDATES] {len(ranked)}")

    return ranked[:25]


# =========================================================
# DOWNLOAD IMAGES
# =========================================================
def download_images(image_urls, idx, title):

    paths = []

    seen_hashes = set()

    print(f"[DOWNLOADING IMAGES] {len(image_urls)} candidates")

    for i, url in enumerate(image_urls):

        # =================================================
        # LIMIT
        # =================================================
        if len(paths) >= MAX_IMAGES_PER_PRODUCT:
            break

        try:

            print(f"[IMAGE] {url}")

            response = session.get(
                url,
                headers=HEADERS,
                timeout=20
            )

            response.raise_for_status()

            # =================================================
            # OPEN IMAGE
            # =================================================
            img = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            # =================================================
            # VALIDATION
            # =================================================
            if not validate_image(img):
                print("[SKIPPED] Invalid size/aspect")
                continue

            # =================================================
            # DEDUPLICATION
            # =================================================
            hash_img = img.resize((128, 128))

            image_hash = hashlib.md5(
                hash_img.tobytes()
            ).hexdigest()

            if image_hash in seen_hashes:
                print("[SKIPPED] Duplicate")
                continue

            seen_hashes.add(image_hash)

            # =================================================
            # RESIZE
            # =================================================
            img.thumbnail((MAX_DIM, MAX_DIM))

            # =================================================
            # SAVE
            # =================================================
            safe_title = re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                title.lower()
            )[:50]

            path = os.path.join(
                IMAGE_DIR,
                f"{safe_title}_{len(paths)}.jpg"
            )

            img.save(
                path,
                "JPEG",
                quality=85,
                optimize=True
            )

            paths.append(path)

            print(f"[SAVED] {path}")

            time.sleep(0.15)

        except Exception as e:

            print(f"[IMAGE ERROR] {url}")
            print(e)

            continue

    # =====================================================
    # FINAL REPORT
    # =====================================================
    print(f"[IMAGES SAVED TOTAL] {len(paths)}")

    return paths


# =========================================================
# CLIP EMBEDDINGS
# =========================================================

def get_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        emb = clip_model.get_image_features(
            **inputs
        )

    emb = emb / emb.norm(
        dim=-1,
        keepdim=True
    )

    return emb.cpu().numpy()[0]


def get_model_embedding(image_paths):

    embeddings = []

    for path in image_paths:

        try:
            emb = get_embedding(path)
            embeddings.append(emb)

        except:
            continue

    if not embeddings:
        return None

    embeddings = np.array(embeddings)

    # ---------------------------------------------
    # Normalize
    final_embedding = embeddings.mean(axis=0)

    final_embedding = (
        final_embedding /
        np.linalg.norm(final_embedding)
    )

    return final_embedding


# =========================================================
# GROUND TRUTH BUILDER
# =========================================================

def save_ground_truth_entry(
    title,
    brand,
    specs,
    source_image
):

    gt_path = os.path.join(
        GROUND_TRUTH_DIR,
        "ground_truth.json"
    )

    data = []

    if os.path.exists(gt_path):

        with open(gt_path, "r") as f:
            data = json.load(f)

    image_name = (
        f"gt_{len(data):03d}.jpg"
    )

    image_dest = os.path.join(
        GROUND_TRUTH_IMAGES,
        image_name
    )

    shutil.copy(
        source_image,
        image_dest
    )

    entry = {
        "image": image_name,
        "title": title,
        "brand": brand,
        "specs": specs
    }

    data.append(entry)

    with open(gt_path, "w") as f:

        json.dump(
            data,
            f,
            indent=2
        )


# MAIN PRODUCT PARSER
# =========================================================
def parse_product_page(url):

    html = fetch_page(url)

    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    # =====================================================
    # TRY JSON-LD FIRST
    # =====================================================

    data_blocks = extract_json_ld(soup)
    data_blocks = flatten_jsonld(data_blocks)

    product_json = find_product_jsonld(data_blocks)

    print(f"[JSONLD FOUND] {product_json is not None}")

    if product_json:
        print(product_json)
    

    # =====================================================
    # USE JSONLD ONLY IF VALID
    # =====================================================

    if (
        product_json and
        isinstance(product_json, dict) and
        product_json.get("name")
    ):

        parsed = parse_jsonld(product_json)

        if parsed["title"].strip():

            print("[USING JSONLD PARSER]")

            all_images = collect_all_images(
                soup,
                url,
                parsed["images"]
            )

            return {
                "url": url,
                "title": parsed["title"],
                "brand": parsed["brand"],
                "raw_text": (
                    parsed["description"]
                    + " "
                    + parsed["title"]
                ),
                "images": all_images,
                "source": "jsonld"
            }
        

        domain = urlparse(url).netloc.lower()
        if "notebookcheck" in domain:

            images = extract_notebookcheck_images(soup)

            return {
                "url": url,
                "title": soup.title.text.strip(),
                "brand": None,
                "raw_text": soup.get_text(" "),
                "images": images,
                "source": "notebookcheck"
            }

    # =====================================================
    # FALLBACK HTML
    # =====================================================

    print("[USING FALLBACK HTML PARSER]")

    fallback = parse_html_fallback(soup, url)

    return {
        "url": url,
        "title": fallback["title"],
        "brand": None,
        "raw_text": fallback["raw_text"],
        "images": fallback["images"],
        "source": "html"
    }


# =========================================================
# DATASET BUILDER
# =========================================================

# detect dupes -------------------------------
def normalize_key(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9]", "", title)
    return title

def build_dataset(urls):

    dataset = []
    embeddings = []

    seen_titles = set()

    for idx, url in enumerate(tqdm(urls)):
        print("\n===================================")
        print(f"[PRODUCT {idx+1}/{len(urls)}]")
        print(url)

        product = parse_product_page(url)
        print("\n[FULL PRODUCT]")
        print(product)

        if product:
            print(f"[PARSED] {product['title']}")
        else:
            print("[PARSE FAILED]")

        if not product:
            continue

        # ---------------------------------------------
        # TITLE CLEANING
        def clean_title(title):

            REMOVE = [
                "review",
                "test",
                "hands-on",
                "- notebookcheck.net reviews",
            ]

            title = re.sub(r"review.*", "", title, flags=re.I)

            for r in REMOVE:
                title = title.replace(r, "")

            title = title.strip(" -|:")

            return title.strip()
        

        title = clean_title(product["title"])

        if not title:
            print("[SKIPPED] Empty title")
            continue

        key = normalize_key(title)
        if key in seen_titles:
            continue

        seen_titles.add(title)

        # ---------------------------------------------
        # DOWNLOAD MULTIPLE IMAGES
        # ---------------------------------------------
        print("[IMAGES] Downloading images...")

        image_paths = download_images(
            product["images"],
            idx,
            title
        )

        print(f"[IMAGES] Downloaded {len(image_paths)} images")


        # ---------------------------------------------
        # PARSE SPECS
        # ---------------------------------------------
        parsed = parse_specs(
            product["raw_text"],
            title
        )

        categorized = categorize(parsed)

        # ---------------------------------------------
        # MULTI-VIEW EMBEDDING
        # ---------------------------------------------
        print("[EMBEDDING] Generating embedding...")
        embedding = get_model_embedding(
            image_paths,
        )
        print("[EMBEDDING OK]")

        if embedding is None:
            continue

        # ---------------------------------------------
        # GROUND TRUTH COLLECTION
        # ---------------------------------------------

        if len(image_paths) > 0:

            gt_path = os.path.join(
                GROUND_TRUTH_DIR,
                "ground_truth.json"
            )

            current_count = 0

            if os.path.exists(gt_path):

                with open(gt_path, "r") as f:
                    current_count = len(
                        json.load(f)
                    )

            if current_count < 20:

                save_ground_truth_entry(
                    title,
                    parsed["brand"],
                    parsed,
                    image_paths[0]
                )

                print(
                    f"[GROUND TRUTH] Added sample {current_count+1}/20"
                )

        # ---------------------------------------------
        # BUILD ENTRY
        # ---------------------------------------------
        entry = {
            "id": idx,
            "title": title,
            "brand": parsed["brand"],
            "images": image_paths,
            "url": product["url"],

            "specs": {
                **parsed,
                **categorized
            },

            #"embedding": embedding.tolist(),

            "source_type": product["source"]
        }

        dataset.append(entry)
        print(f"[SUCCESS] Added: {title}")

        embeddings.append(embedding)

        time.sleep(2)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    print("\n===================================")
    print(f"FINAL PRODUCTS: {len(dataset)}")
    print(f"FINAL EMBEDDINGS: {len(embeddings)}")

    return dataset, embeddings


# =========================================================
# FAISS INDEX TODO
# =========================================================

# def build_faiss(embeddings):
#
#     index = faiss.IndexFlatIP(
#         embeddings.shape[1]
#     )
#
#     index.add(embeddings)
#
#     return index


# =========================================================
# SAVE
# =========================================================

def save_all(dataset, embeddings):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    database_path = os.path.join(
        OUTPUT_DIR,
        "database.json"
    )

    # =====================================================
    # LOAD OLD DATA
    # =====================================================
    old_data = []

    if os.path.exists(database_path):

        try:
            with open(database_path, "r") as f:
                old_data = json.load(f)

        except:
            old_data = []

    # =====================================================
    # MERGE
    # =====================================================
    merged = old_data + dataset

    # Remove duplicates by title
    unique = {}

    for item in merged:
        unique[item["title"]] = item

    final_data = list(unique.values())

    # =====================================================
    # SAVE DATABASE
    # =====================================================
    with open(database_path, "w") as f:

        json.dump(
            final_data,
            f,
            indent=2
        )

    # =====================================================
    # SAVE EMBEDDINGS
    # =====================================================
    embedding_path = os.path.join(
        OUTPUT_DIR,
        "embeddings.npy"
    )

    if os.path.exists(embedding_path):

        old_embeddings = np.load(embedding_path)

        embeddings = np.concatenate(
            [old_embeddings, embeddings],
            axis=0
        )

    np.save(
        embedding_path,
        embeddings
    )

    print(f"[TOTAL DATASET] {len(final_data)}")


# GETS MULTIPLE PRODUCT LINKS ======================================
def get_product_links(category_url):

    print(f"[CRAWLER] Fetching: {category_url}")

    html = fetch_page(category_url)

    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    links = set()

    PRODUCT_KEYWORDS = [
        "laptop",
        "notebook",
        "legion",
        "thinkpad",
        "zenbook",
        "vivobook",
        "rog",
        "predator",
        "omen",
        "xps",
        "macbook"
    ]

    BAD_KEYWORDS = [
        "accessory",
        "support",
        "driver",
        "forum",
        "news",
        "blog",
        "cart",
        "login"
    ]

    for a in soup.find_all("a"):

        href = a.get("href")

        if not href:
            continue

        href = urljoin(category_url, href)

        lower = href.lower()

        # ==========================================
        # PRODUCT FILTER
        # ==========================================
        if not any(
            k in lower
            for k in PRODUCT_KEYWORDS
        ):
            continue

        # ==========================================
        # BAD FILTER
        # ==========================================
        if any(
            b in lower
            for b in BAD_KEYWORDS
        ):
            continue

        # ==========================================
        # ONLY VALID PAGES
        # ==========================================
        if href.startswith("http"):
            links.add(href)

    print(f"[CRAWLER] Found {len(links)} links")

    return list(links)



# NOTEBOOKCHECK SCRAPER
# =========================================================
def get_notebookcheck_links(
    start_page=0,
    end_page=10,
    max_links=500
):

    links = set()

    LAPTOP_BRANDS = [
        "lenovo",
        "asus",
        "acer",
        "msi",
        "hp",
        "dell",
        "razer",
        "gigabyte",
        "apple",
        "huawei",
        "samsung",
        "framework",
        "lg",
    ]

    BAD_TERMS = [
        "smartphone",
        "tablet",
        "monitor",
        "gpu",
        "graphics-card",
        "headphones",
        "earbuds",
        "camera",
        "mini-pc",
        "desktop",
        "server",
    ]

    for page in range(start_page, end_page + 1):

        # =====================================
        # NOTEBOOKCHECK PAGINATION
        # =====================================

        if page == 0:
            url = "https://www.notebookcheck.net/Reviews.55.0.html"
        else:
            url = f"https://www.notebookcheck.net/Reviews.55.0.html?&ns_page={page}"

        print("\n=====================================")
        print(f"[PAGE {page}] {url}")

        html = fetch_page(url)

        if not html:
            continue

        soup = BeautifulSoup(html, "lxml")

        page_links = 0

        for a in soup.find_all("a"):

            href = a.get("href")

            if not href:
                continue

            href = urljoin(url, href)

            lower = href.lower()

            # =====================================
            # MUST BE REVIEW PAGE
            # =====================================

            if "review" not in lower:
                continue

            if not lower.endswith(".html"):
                continue

            # =====================================
            # MUST CONTAIN LAPTOP BRANDS
            # =====================================

            if not any(
                brand in lower
                for brand in LAPTOP_BRANDS
            ):
                continue

            # =====================================
            # FILTER BAD DEVICES
            # =====================================

            if any(
                bad in lower
                for bad in BAD_TERMS
            ):
                continue

            # =====================================
            # ADD LINK
            # =====================================

            if href not in links:
                links.add(href)
                page_links += 1

                print(f"[FOUND] {href}")

            # =====================================
            # LIMIT
            # =====================================

            if len(links) >= max_links:
                break

        print(f"[PAGE LINKS] {page_links}")
        print(f"[TOTAL LINKS] {len(links)}")

        if len(links) >= max_links:
            break

        time.sleep(1)

    print("\n=====================================")
    print(f"[FINAL TOTAL LINKS] {len(links)}")

    return list(links)


# MAIN
# =========================================================

if __name__ == "__main__":

    # MANUAL URLS
        #"https://www.lenovo.com/us/en/laptops/"
        #"https://www.asus.com/laptops/for-home/all-series"
        #"https://www.dell.com/en-us/lp/sitemap"
        #"https://www.msi.com/Laptops"

    #urls = [ 
    #    "https://www.notebookcheck.net/Lenovo-Legion-5i-16-G9-review-The-fast-gaming-laptop-with-Raptor-Lake-HX-and-an-AI-engine.798236.0.html"
    #]
    
    #urls = get_product_links(
    #    "https://www.asus.com/laptops/for-home/all-series/"
    #     max_links=10
    #)

    # NOTEBOOKCHECK -----------------------------------------------
    urls = get_notebookcheck_links(
        start_page=0,
        end_page=500,
        max_links=1000 # 100+ pages, 1000 links
    )

    urls = list(set(urls)) # remov duplic urls

    print("\n[URLS FOUND]")
    for u in urls:
        print(u)

    dataset, embeddings = build_dataset(urls)

    if len(embeddings) == 0:
        print("No embeddings created")
        exit()

    save_all(
        dataset,
        embeddings
    )

    print(f"Saved {len(dataset)} products")
