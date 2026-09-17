import pandas as pd

CSV_PATH = "data/labeled/clean_dataset.csv"


def main():
    print("=" * 60)
    print("CropSense - Dataset Quality Analysis")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    print(f"\nTotal labelled images: {len(df)}")

    # --------------------------------------------------
    # 1. Missing values
    # --------------------------------------------------
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # --------------------------------------------------
    # 2. Duplicate images
    # --------------------------------------------------
    duplicate_images = df["image_filename"].duplicated().sum()

    print("\nDuplicate image filenames:")
    print(duplicate_images)

    # --------------------------------------------------
    # 3. Duplicate image paths
    # --------------------------------------------------
    duplicate_paths = df["image_path"].duplicated().sum()

    print("\nDuplicate image paths:")
    print(duplicate_paths)

    # --------------------------------------------------
    # 4. Crop distribution
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("CROP DISTRIBUTION")
    print("-" * 60)

    crop_counts = df["crop"].value_counts()

    print(crop_counts.to_string())

    print("\nCrop percentages:")
    print(
        (crop_counts / len(df) * 100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    # --------------------------------------------------
    # 5. Growth-stage distribution
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("GROWTH-STAGE DISTRIBUTION")
    print("-" * 60)

    stage_counts = df["growth_stage"].value_counts()

    print(stage_counts.to_string())

    print("\nStage percentages:")
    print(
        (stage_counts / len(df) * 100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    # --------------------------------------------------
    # 6. Crop + stage combinations
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("CROP + GROWTH-STAGE COMBINATIONS")
    print("-" * 60)

    combinations = (
        df.groupby(["crop", "growth_stage"])
        .size()
        .reset_index(name="images")
        .sort_values("images", ascending=False)
    )

    print(combinations.to_string(index=False))

    # --------------------------------------------------
    # 7. Very rare crop classes
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("RARE CROP CLASSES (< 10 images)")
    print("-" * 60)

    rare_crops = crop_counts[crop_counts < 10]

    if len(rare_crops) == 0:
        print("None")
    else:
        print(rare_crops.to_string())

    # --------------------------------------------------
    # 8. Very rare growth stages
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("RARE GROWTH STAGES (< 20 images)")
    print("-" * 60)

    rare_stages = stage_counts[stage_counts < 20]

    if len(rare_stages) == 0:
        print("None")
    else:
        print(rare_stages.to_string())

    # --------------------------------------------------
    # 9. Save distributions
    # --------------------------------------------------
    crop_counts.rename("image_count").to_csv(
        "data/labeled/crop_distribution.csv"
    )

    stage_counts.rename("image_count").to_csv(
        "data/labeled/stage_distribution.csv"
    )

    combinations.to_csv(
        "data/labeled/crop_stage_distribution.csv",
        index=False
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS COMPLETE")
    print("=" * 60)

    print("Generated:")
    print("  data/labeled/crop_distribution.csv")
    print("  data/labeled/stage_distribution.csv")
    print("  data/labeled/crop_stage_distribution.csv")


if __name__ == "__main__":
    main()
