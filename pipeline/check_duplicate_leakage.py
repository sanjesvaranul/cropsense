import os
import hashlib
import pandas as pd

IMAGE_DIR = "data/raw/images"

TRAIN_PATH = "data/labeled/train.csv"
VAL_PATH = "data/labeled/val.csv"


def image_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def main():
    print("=" * 60)
    print("CropSense - Duplicate Leakage Check")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    print(f"\nTraining images   : {len(train)}")
    print(f"Validation images : {len(val)}")

    train_hashes = {}
    val_hashes = {}

    # Hash training images
    for filename in train["image_filename"]:
        path = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(path):
            h = image_hash(path)
            train_hashes.setdefault(h, []).append(filename)

    # Hash validation images
    for filename in val["image_filename"]:
        path = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(path):
            h = image_hash(path)
            val_hashes.setdefault(h, []).append(filename)

    # Find hashes appearing in both sets
    overlap = set(train_hashes) & set(val_hashes)

    print(f"\nUnique training image hashes   : {len(train_hashes)}")
    print(f"Unique validation image hashes : {len(val_hashes)}")
    print(f"Duplicate hashes across splits : {len(overlap)}")

    if len(overlap) == 0:
        print("\nPASS: No duplicate-image leakage detected.")
    else:
        print("\nWARNING: Duplicate-image leakage detected!")

        print("\nExamples:")

        for h in list(overlap)[:10]:
            print("\nTraining:")
            for f in train_hashes[h]:
                print(" ", f)

            print("Validation:")
            for f in val_hashes[h]:
                print(" ", f)

    print("\n" + "=" * 60)
    print("Duplicate leakage check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
