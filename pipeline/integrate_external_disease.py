import os
import io
import hashlib
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image

EXTERNAL_DIR = Path("data/external/rice_diseases")
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

REPOS = [
    {
        "name": "sandragirish",
        "url": "https://raw.githubusercontent.com/sandragirish/Rice_plant_disease_detection/main/rice-leaf.zip",
        "class_mapping": {
            "bacterial_leaf_blight": "Bacterial Leaf Blight",
            "blast": "Leaf Blast",
            "brownspot": "Brown Spot"
        }
    },
    {
        "name": "MHassaanButt",
        "url": "https://github.com/MHassaanButt/Rice-Disease-Classfication/archive/refs/heads/master.zip",
        "class_mapping": {
            "Bacterial leaf blight": "Bacterial Leaf Blight",
            "Brown spot": "Brown Spot",
            "Leaf smut": "Leaf Smut"
        }
    }
]


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def process_and_extract():
    print("=" * 60)
    print("CropSense - External Rice Disease Dataset Integration")
    print("=" * 60)

    seen_hashes = set()
    category_counts = {}

    for repo in REPOS:
        print(f"\nProcessing source: {repo['name']}...")
        print(f"Downloading from: {repo['url']}")
        req = urllib.request.Request(repo["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            zip_bytes = resp.read()
        print(f"Downloaded {len(zip_bytes)} bytes.")

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                ext = os.path.splitext(member)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                # Match class
                matched_category = None
                for raw_name, norm_name in repo["class_mapping"].items():
                    if f"/{raw_name}/" in member.replace("\\", "/") or f"/{raw_name.lower()}/" in member.lower().replace("\\", "/"):
                        matched_category = norm_name
                        break

                if not matched_category:
                    continue

                img_data = zf.read(member)
                h = file_hash(img_data)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Verify valid image
                try:
                    with Image.open(io.BytesIO(img_data)) as img:
                        img.verify()
                except Exception as e:
                    print(f"Skipping corrupt image {member}: {e}")
                    continue

                category_dir = EXTERNAL_DIR / matched_category
                category_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{repo['name']}_{h[:8]}{ext}"
                out_path = category_dir / filename
                with open(out_path, "wb") as f:
                    f.write(img_data)

                category_counts[matched_category] = category_counts.get(matched_category, 0) + 1

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:25s}: {count:4d} images")
    total_images = sum(category_counts.values())
    print(f"  {'TOTAL EXTERNAL DISEASE':25s}: {total_images:4d} images")
    print(f"Saved into: {EXTERNAL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    process_and_extract()
