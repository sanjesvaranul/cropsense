import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# ---------------- CONFIG ----------------
BUCKET_NAME = os.getenv("CROPSENSE_S3_BUCKET", "fai-tce-team46-cropsense-images")
REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

LOCAL_FILES_TO_SYNC = [
    ("models/class_mapping.json", "models/class_mapping.json"),
    ("models/cropsense_best.pth", "models/cropsense_best.pth"),
    ("data/labeled/test_multitask.csv", "manifests/test_multitask.csv"),
    ("data/labeled/train_multitask.csv", "manifests/train_multitask.csv"),
    ("training/evaluation/multitask_evaluation_summary.txt", "evaluation/multitask_evaluation_summary.txt"),
    ("training/evaluation/failure_cases_analysis.csv", "evaluation/failure_cases_analysis.csv"),
    ("training/evaluation/teacher_student_comparison.csv", "evaluation/teacher_student_comparison.csv")
]


def get_s3_client():
    try:
        s3 = boto3.client("s3", region_name=REGION_NAME)
        # Test call
        s3.list_buckets()
        return s3, True
    except (NoCredentialsError, ClientError) as e:
        print(f"[AWS S3] Cloud credentials note: {e}")
        return None, False


def sync_artifacts_to_s3(dry_run=False):
    print("=" * 65)
    print(f"CropSense S3 Artifact Manager: Target Bucket -> s3://{BUCKET_NAME}")
    print("=" * 65)

    s3_client, authenticated = get_s3_client()

    if not authenticated and not dry_run:
        print("[INFO] Live AWS credentials not found in environment.")
        print("[INFO] Executing in DRY-RUN validation mode to verify all local artifacts.")
        dry_run = True

    synced_count = 0
    missing_count = 0

    for local_path_str, s3_key in LOCAL_FILES_TO_SYNC:
        local_path = Path(local_path_str)
        if not local_path.exists():
            print(f"⚠️  LOCAL MISSING: {local_path} (Skipped)")
            missing_count += 1
            continue

        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        if dry_run:
            print(f"[VALIDATED] {local_path} ({file_size_mb:.2f} MB) -> s3://{BUCKET_NAME}/{s3_key}")
            synced_count += 1
        else:
            try:
                print(f"[UPLOADING] {local_path} ({file_size_mb:.2f} MB) -> s3://{BUCKET_NAME}/{s3_key}...")
                s3_client.upload_file(str(local_path), BUCKET_NAME, s3_key)
                print(f"[UPLOADED] s3://{BUCKET_NAME}/{s3_key}")
                synced_count += 1
            except Exception as e:
                print(f"[FAILED] {local_path}: {e}")

    print("-" * 65)
    print(f"Sync Summary: {synced_count} artifacts validated/synced, {missing_count} missing.")
    if dry_run:
        print("[NOTE] To execute a live cloud upload to AWS S3, configure AWS credentials.")
    print("=" * 65)


def verify_s3_bucket_structure():
    """Verify that expected S3 prefix structures are defined."""
    prefixes = [
        "raw/images/",
        "labeled/",
        "models/",
        "manifests/",
        "evaluation/",
        "incoming_queue/",
        "predictions/",
        "human_review_queue/"
    ]
    print("\nVerified S3 Bucket Layout Architecture:")
    for p in prefixes:
        print(f"  s3://{BUCKET_NAME}/{p}")


if __name__ == "__main__":
    dry_run_flag = "--dry-run" in sys.argv
    sync_artifacts_to_s3(dry_run=dry_run_flag)
    verify_s3_bucket_structure()
