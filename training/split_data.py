import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/labeled/clean_dataset_fixed.csv"
TRAIN_PATH = "data/labeled/train.csv"
VAL_PATH = "data/labeled/val.csv"
TEST_PATH = "data/labeled/test.csv"

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("CropSense - Train / Validation / Test Split")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    required = ["image_filename", "image_path", "crop", "growth_stage"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove duplicate image records
    df = df.drop_duplicates(subset=["image_filename"]).reset_index(drop=True)

    print(f"\nTotal unique images: {len(df)}")

    # Reserve one image from every crop class for training.
    # This ensures even singleton classes, such as Mango, are represented.
    anchors = (
        df.groupby("crop", group_keys=False)
        .sample(n=1, random_state=RANDOM_STATE)
    )

    remaining = df.drop(index=anchors.index)

    # Split remaining images: 80% train, 20% temporary
    train_rest, temp = train_test_split(
        remaining,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Split temporary set equally into validation and test
    val_df, test_df = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Add reserved crop examples to training
    train_df = pd.concat([train_rest, anchors]).sample(
        frac=1, random_state=RANDOM_STATE
    )

    # Save split files
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nSPLIT RESULT")
    print("-" * 60)
    print(f"Training images   : {len(train_df)}")
    print(f"Validation images : {len(val_df)}")
    print(f"Testing images    : {len(test_df)}")

    # Verify no filename overlap
    train_names = set(train_df["image_filename"])
    val_names = set(val_df["image_filename"])
    test_names = set(test_df["image_filename"])

    overlap = (
        (train_names & val_names)
        | (train_names & test_names)
        | (val_names & test_names)
    )

    print(f"\nFilename overlap: {len(overlap)}")
    print("PASS: No filename overlap." if not overlap else "ERROR: Overlap found!")

    # Verify every crop class is represented in training
    all_crops = set(df["crop"].dropna().unique())
    train_crops = set(train_df["crop"].dropna().unique())
    missing_crops = all_crops - train_crops

    print(f"\nCrop classes in full dataset : {len(all_crops)}")
    print(f"Crop classes in training     : {len(train_crops)}")
    print(f"Crop classes missing in train: {sorted(missing_crops)}")

    print("\nTraining crop distribution:")
    print(train_df["crop"].value_counts().to_string())

    print("\nValidation crop distribution:")
    print(val_df["crop"].value_counts().to_string())

    print("\nTesting crop distribution:")
    print(test_df["crop"].value_counts().to_string())

    print("\nSaved files:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    if overlap or missing_crops:
        raise ValueError("Split verification failed.")

    print("\nSplit completed and verified.")


if __name__ == "__main__":
    main()
