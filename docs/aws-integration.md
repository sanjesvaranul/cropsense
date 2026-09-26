# CropSense AWS Integration

AWS S3 and AWS Lambda are used for cloud-based dataset storage and verification.

## S3

Bucket: `fai-tce-team46-cropsense-images`

- `labeled/clean_dataset_fixed.csv` — labeled dataset CSV
- `raw/images/` — raw crop images
- 3,422 image objects were verified under `raw/images/`.

## Lambda

Function: `fai-tce-team46-cropsense-api`

- Runtime: Python 3.12
- Handler: `lambda_function.lambda_handler`
- Memory: 256 MB
- Timeout: 30 seconds
- Architecture: x86_64
- Region: `ap-south-1`

## Verification

The deployed Lambda successfully verified the S3 dataset.

- CSV rows: 3,165
- Unique CSV images: 3,165
- S3 images: 3,422
- Images found: 3,165
- Images missing: 0
- Status: ALL FOUND

## Source

Lambda source: `aws/lambda_function.py`

The implementation was introduced in commit `5ee794c` — Add AWS Lambda S3 image verification.
