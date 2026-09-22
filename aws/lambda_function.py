
import json
import boto3
import csv
import io

s3 = boto3.client("s3")

BUCKET_NAME = "fai-tce-team46-cropsense-images"
CSV_KEY = "labeled/clean_dataset_fixed.csv"
IMAGE_PREFIX = "raw/images/"


def lambda_handler(event, context):
    try:
        # 1. Read the CSV file from S3
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=CSV_KEY
        )

        csv_content = response["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # 2. Collect image filenames from the CSV
        csv_filenames = set()

        for row in rows:
            path = row["image_path"].replace("\\", "/")
            filename = path.split("/")[-1]
            csv_filenames.add(filename)

        # 3. List S3 image filenames (paginated)
        s3_filenames = set()

        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=BUCKET_NAME,
            Prefix=IMAGE_PREFIX
        ):
            for obj in page.get("Contents", []):
                filename = obj["Key"][len(IMAGE_PREFIX):]

                # Ignore nested folders
                if filename and "/" not in filename:
                    s3_filenames.add(filename)

        # 4. Compare CSV filenames with S3 filenames
        found = csv_filenames & s3_filenames
        missing = csv_filenames - s3_filenames

        return {
            "statusCode": 200,
            "body": json.dumps({
                "csv_rows": len(rows),
                "unique_csv_images": len(csv_filenames),
                "s3_images": len(s3_filenames),
                "images_found": len(found),
                "images_missing": len(missing),
                "missing_samples": sorted(list(missing))[:10],
                "verification": (
                    "ALL FOUND" if not missing
                    else "MISSING IMAGES"
                )
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }