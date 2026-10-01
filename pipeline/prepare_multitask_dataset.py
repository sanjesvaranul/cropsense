import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ORIGINAL_CLEAN_CSV = Path("data/labeled/clean_dataset_fixed.csv")
ORIGINAL_TRAIN_CSV = Path("data/labeled/train.csv")
ORIGINAL_VAL_CSV = Path("data/labeled/val.csv")
ORIGINAL_TEST_CSV = Path("data/labeled/test.csv")

EXTERNAL_DIR = Path("data/external/rice_diseases")

OUT_TRAIN_CSV = Path("data/labeled/train_multitask.csv")
OUT_VAL_CSV = Path("data/labeled/val_multitask.csv")
OUT_TEST_CSV = Path("data/labeled/test_multitask.csv")
OUT_MAPPING_PATH = Path("models/class_mapping.json")

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("CropSense - Prepare Multi-Task Dataset (Crop + Stage + Disease)")
    print("=" * 60)

    # 1. Load existing splits and attach default condition = "Healthy"
    train_orig = pd.read_csv(ORIGINAL_TRAIN_CSV)
    val_orig = pd.read_csv(ORIGINAL_VAL_CSV)
    test_orig = pd.read_csv(ORIGINAL_TEST_CSV)

    train_orig["condition"] = "Healthy"
    val_orig["condition"] = "Healthy"
    test_orig["condition"] = "Healthy"

    # 2. Collect external disease images
    external_rows = []
    for disease_folder in sorted(EXTERNAL_DIR.iterdir()):
        if not disease_folder.is_dir():
            continue
        disease_name = disease_folder.name
        for img_file in disease_folder.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                external_rows.append({
                    "image_filename": img_file.name,
                    "image_path": str(img_file).replace("\\", "/"),
                    "crop": "Rice (Paddy)",
                    "growth_stage": "vegetation",
                    "condition": disease_name
                })

    ext_df = pd.DataFrame(external_rows)
    print(f"\nExternal disease records found: {len(ext_df)}")
    print(ext_df["condition"].value_counts())

    # 3. Stratified split of external disease images (80% train, 10% val, 10% test)
    ext_train, ext_temp = train_test_split(
        ext_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=ext_df["condition"]
    )
    ext_val, ext_test = train_test_split(
        ext_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=ext_temp["condition"]
    )

    print(f"External Split -> Train: {len(ext_train)}, Val: {len(ext_val)}, Test: {len(ext_test)}")

    # 4. Merge original splits with external disease splits
    train_full = pd.concat([train_orig, ext_train], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    val_full = pd.concat([val_orig, ext_val], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    test_full = pd.concat([test_orig, ext_test], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # 5. Extract class mappings
    crop_classes = sorted(train_full["crop"].unique())
    stage_classes = sorted(train_full["growth_stage"].unique())
    condition_classes = sorted(train_full["condition"].unique())

    # Ensure "Healthy" is index 0 for intuitive baseline
    if "Healthy" in condition_classes:
        condition_classes.remove("Healthy")
        condition_classes = ["Healthy"] + condition_classes

    crop_map = {name: idx for idx, name in enumerate(crop_classes)}
    stage_map = {name: idx for idx, name in enumerate(stage_classes)}
    condition_map = {name: idx for idx, name in enumerate(condition_classes)}

    mapping_data = {
        "crop_map": crop_map,
        "stage_map": stage_map,
        "condition_map": condition_map
    }

    OUT_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MAPPING_PATH, "w") as f:
        json.dump(mapping_data, f, indent=2)

    # 6. Save merged datasets
    train_full.to_csv(OUT_TRAIN_CSV, index=False)
    val_full.to_csv(OUT_VAL_CSV, index=False)
    test_full.to_csv(OUT_TEST_CSV, index=False)

    print("\n" + "=" * 60)
    print("MULTI-TASK DATASET SUMMARY")
    print("=" * 60)
    print(f"Train samples: {len(train_full)}")
    print(f"Val samples  : {len(val_full)}")
    print(f"Test samples : {len(test_full)}")
    print(f"\nCrop Map ({len(crop_map)} classes):")
    for k, v in crop_map.items():
        print(f"  {v:2d}: {k}")
    print(f"\nStage Map ({len(stage_map)} classes):")
    for k, v in stage_map.items():
        print(f"  {v:2d}: {k}")
    print(f"\nCondition Map ({len(condition_map)} classes):")
    for k, v in condition_map.items():
        print(f"  {v:2d}: {k}")
    print(f"\nCondition distribution in Train:\n{train_full['condition'].value_counts()}")
    print(f"\nCondition distribution in Val:\n{val_full['condition'].value_counts()}")
    print(f"\nCondition distribution in Test:\n{test_full['condition'].value_counts()}")
    print("Saved files:")
    print(f"  {OUT_TRAIN_CSV}")
    print(f"  {OUT_VAL_CSV}")
    print(f"  {OUT_TEST_CSV}")
    print(f"  {OUT_MAPPING_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
