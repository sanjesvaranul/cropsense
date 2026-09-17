import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = "data/labeled/clean_dataset.csv"

TRAIN_PATH = "data/labeled/train.csv"
VAL_PATH = "data/labeled/val.csv"


def main():
    print("=" * 60)
    print("CropSense - Group-Aware Train/Validation Split")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal images: {len(df)}")
    print(f"Unique image groups: {df['image_path'].nunique()}")

    # Group by image_path so related images stay in the same split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["image_path"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Save
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)

    print("\n" + "-" * 60)
    print("SPLIT RESULT")
    print("-" * 60)

    print(f"Training images   : {len(train_df)}")
    print(f"Validation images : {len(val_df)}")

    print(
        f"Training groups   : {train_df['image_path'].nunique()}"
    )

    print(
        f"Validation groups : {val_df['image_path'].nunique()}"
    )

    # Verify no group leakage
    train_groups = set(train_df["image_path"])
    val_groups = set(val_df["image_path"])

    overlap = train_groups.intersection(val_groups)

    print(f"\nGroup overlap     : {len(overlap)}")

    if len(overlap) == 0:
        print("PASS: No group leakage detected.")
    else:
        print("WARNING: Group leakage detected!")

    print("\nTraining crop distribution:")
    print(train_df["crop"].value_counts().to_string())

    print("\nValidation crop distribution:")
    print(val_df["crop"].value_counts().to_string())

    print("\nTraining stage distribution:")
    print(train_df["growth_stage"].value_counts().to_string())

    print("\nValidation stage distribution:")
    print(val_df["growth_stage"].value_counts().to_string())

    print("\nSaved:")
    print(f"  {TRAIN_PATH}")
    print(f"  {VAL_PATH}")

    print("\n" + "=" * 60)
    print("Split complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
