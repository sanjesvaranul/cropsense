import os
import pandas as pd

IMAGE_DIR = "data/raw/images"
CSV_PATH = "data/labeled/Ground_truth_Points.csv"


def main():
    print("=" * 60)
    print("CropSense Dataset Validation")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Check image files
    # -------------------------------------------------
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    image_stems = {
        os.path.splitext(f)[0]
        for f in image_files
    }

    print(f"\nImages found              : {len(image_files)}")

    # -------------------------------------------------
    # 2. Load ground-truth CSV
    # -------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    print(f"GT CSV rows               : {len(df)}")
    print(f"Unique image_path values  : {df['image_path'].nunique()}")

    # -------------------------------------------------
    # 3. Required columns
    # -------------------------------------------------
    required_columns = [
        "image_path",
        "final_crop_name",
        "final_crop_stage",
    ]

    print("\nRequired columns:")

    for col in required_columns:
        if col in df.columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [MISSING] {col}")

    # -------------------------------------------------
    # 4. Missing values
    # -------------------------------------------------
    print("\nMissing values:")

    for col in required_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].isna().sum()}")

    # -------------------------------------------------
    # 5. Duplicate GT records
    # -------------------------------------------------
    duplicate_rows = df.duplicated().sum()

    print(f"\nDuplicate complete GT rows : {duplicate_rows}")

    # -------------------------------------------------
    # 6. Label distributions
    # -------------------------------------------------
    print("\nCrop distribution:")
    print(df["final_crop_name"].value_counts().to_string())

    print("\nGrowth-stage distribution:")
    print(df["final_crop_stage"].value_counts().to_string())

    # -------------------------------------------------
    # 7. Check label consistency per image_path
    # -------------------------------------------------
    grouped = df.groupby("image_path").agg(
        crop_count=("final_crop_name", "nunique"),
        stage_count=("final_crop_stage", "nunique"),
    )

    conflicting_crop = (grouped["crop_count"] > 1).sum()
    conflicting_stage = (grouped["stage_count"] > 1).sum()

    print("\nLabel consistency:")
    print(f"  Images with conflicting crop labels : {conflicting_crop}")
    print(f"  Images with conflicting stage labels: {conflicting_stage}")

    # -------------------------------------------------
    # 8. Direct filename matching
    # -------------------------------------------------
    csv_paths = set(
        df["image_path"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    exact_matches = csv_paths.intersection(image_stems)

    print("\nImage ↔ GT filename matching:")
    print(f"  Exact matches             : {len(exact_matches)}")
    print(f"  CSV unique image IDs     : {len(csv_paths)}")
    print(f"  Local image files        : {len(image_files)}")

    # -------------------------------------------------
    # 9. Save basic validation report
    # -------------------------------------------------
    report = pd.DataFrame({
        "metric": [
            "local_images",
            "gt_rows",
            "unique_gt_image_paths",
            "exact_filename_matches",
            "duplicate_gt_rows",
            "conflicting_crop_labels",
            "conflicting_stage_labels",
        ],
        "value": [
            len(image_files),
            len(df),
            df["image_path"].nunique(),
            len(exact_matches),
            duplicate_rows,
            conflicting_crop,
            conflicting_stage,
        ],
    })

    os.makedirs("data/labeled", exist_ok=True)
    report.to_csv(
        "data/labeled/validation_report.csv",
        index=False
    )

    print("\nValidation report saved to:")
    print("  data/labeled/validation_report.csv")

    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
