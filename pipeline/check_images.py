import os
import hashlib
from PIL import Image

IMAGE_DIR = "data/raw/images"


def file_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def main():
    print("=" * 60)
    print("CropSense - Image Integrity Check")
    print("=" * 60)

    files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    valid = 0
    corrupted = []
    small_images = []
    hashes = {}

    widths = []
    heights = []

    for filename in files:
        path = os.path.join(IMAGE_DIR, filename)

        try:
            with Image.open(path) as img:
                img.verify()

            # Reopen after verify
            with Image.open(path) as img:
                width, height = img.size

            valid += 1
            widths.append(width)
            heights.append(height)

            if width < 100 or height < 100:
                small_images.append(
                    (filename, width, height)
                )

            digest = file_hash(path)

            if digest in hashes:
                hashes[digest].append(filename)
            else:
                hashes[digest] = [filename]

        except Exception as e:
            corrupted.append(
                (filename, str(e))
            )

    duplicate_groups = [
        names for names in hashes.values()
        if len(names) > 1
    ]

    print(f"\nTotal images       : {len(files)}")
    print(f"Valid images       : {valid}")
    print(f"Corrupted images   : {len(corrupted)}")
    print(f"Small images       : {len(small_images)}")
    print(f"Duplicate groups   : {len(duplicate_groups)}")

    if widths:
        print(f"\nMinimum width      : {min(widths)}")
        print(f"Maximum width      : {max(widths)}")
        print(f"Minimum height     : {min(heights)}")
        print(f"Maximum height     : {max(heights)}")

    if corrupted:
        print("\nCorrupted examples:")
        for item in corrupted[:10]:
            print(item)

    if small_images:
        print("\nSmall image examples:")
        for item in small_images[:10]:
            print(item)

    if duplicate_groups:
        print("\nDuplicate examples:")
        for group in duplicate_groups[:5]:
            print(group)

    print("\n" + "=" * 60)
    print("Image integrity check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
