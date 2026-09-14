import os
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def inventory_images(image_dir="data/raw"):
    total = 0
    valid = 0
    invalid = 0
    widths = []
    heights = []

    for root, _, files in os.walk(image_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()

            if ext not in IMAGE_EXTENSIONS:
                continue

            total += 1
            path = os.path.join(root, filename)

            try:
                with Image.open(path) as img:
                    img.verify()

                with Image.open(path) as img:
                    width, height = img.size

                widths.append(width)
                heights.append(height)
                valid += 1

            except Exception as e:
                invalid += 1
                print(f"Invalid image: {path} -> {e}")

    print(f"Total images: {total}")
    print(f"Valid images: {valid}")
    print(f"Invalid images: {invalid}")

    if valid:
        print("Resolution range:")
        print(f"  Width : {min(widths)} - {max(widths)}")
        print(f"  Height: {min(heights)} - {max(heights)}")


if __name__ == "__main__":
    inventory_images()