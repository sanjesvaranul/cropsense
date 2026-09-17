import os
import pandas as pd

IMAGE_DIR = "data/raw/images"
GT_PATH = "data/labeled/Ground_truth_Points.csv"

OUTPUT_PATH = "data/labeled/clean_dataset.csv"
UNMATCHED_PATH = "data/labeled/unmatched_images.csv"


def main():
    print("=" * 60)
    print("CropSense - Creating Clean Dataset")
    print("=" * 60)

    # Load ground truth
    df = pd.read_csv(GT_PATH)

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Clean image_path
    df["image_path"] = (
        df["image_path"]
        .astype(str)
        .str.strip()
    )

    # Get image files
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"\nGround-truth rows after deduplication: {len(df)}")
    print(f"Local image files: {len(image_files)}")

    # ---------------------------------------------------------
    # Build GT lookup
    # ---------------------------------------------------------
    gt_paths = df["image_path"].dropna().unique()

    clean_rows = []
    unmatched_images = []

    for filename in image_files:

        filename_lower = filename.lower()

        matches = [
            path
            for path in gt_paths
            if path.lower() in filename_lower
        ]

        # No GT match
        if len(matches) == 0:
            unmatched_images.append({
                "image_filename": filename
            })
            continue

        # More than one GT match
        if len(matches) > 1:
            print("\nWARNING: Multiple GT paths for image:")
            print(filename)
            print(matches)
            continue

        image_path = matches[0]

        records = df[
            df["image_path"] == image_path
        ]

        # Check crop consistency
        crops = records["final_crop_name"].dropna().unique()

        # Check stage consistency
        stages = records["final_crop_stage"].dropna().unique()

        if len(crops) != 1 or len(stages) != 1:
            print("\nWARNING: Conflicting labels:")
            print(filename)
            print("Crops:", crops)
            print("Stages:", stages)
            continue

        clean_rows.append({
            "image_filename": filename,
            "image_path": image_path,
            "crop": crops[0],
            "growth_stage": stages[0]
        })

    # ---------------------------------------------------------
    # Create clean dataframe
    # ---------------------------------------------------------
    clean_df = pd.DataFrame(clean_rows)

    clean_df = clean_df.drop_duplicates(
        subset=["image_filename"]
    )

    # Save
    clean_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    pd.DataFrame(unmatched_images).to_csv(
        UNMATCHED_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(f"Total local images      : {len(image_files)}")
    print(f"Matched labelled images : {len(clean_df)}")
    print(f"Unmatched images        : {len(unmatched_images)}")

    print("\nCrop distribution:")
    print(
        clean_df["crop"]
        .value_counts()
        .to_string()
    )

    print("\nGrowth-stage distribution:")
    print(
        clean_df["growth_stage"]
        .value_counts()
        .to_string()
    )

    print("\nSaved files:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {UNMATCHED_PATH}")

    print("\n" + "=" * 60)
    print("Clean dataset creation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
