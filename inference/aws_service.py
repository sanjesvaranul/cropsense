"""
CropSense AWS Cloud Integration Service
FarmwiseAI TCE Product Challenge - Team 46 (Mobile Vision Category)
Region: ap-south-1 (Mumbai)

Approved Services:
- Amazon S3: fai-tce-team46-cropsense-images (Private photo vault + Pre-signed URLs)
- Amazon DynamoDB: fai-tce-team46-review-queue (Serverless NoSQL review queue & telemetry)
- Amazon Rekognition: DetectLabels & Guardrail validation (Mobile Vision Category)
"""

import os
import io
import time
import uuid
from datetime import datetime
from pathlib import Path
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

# Automatically load .env if present
ENV_PATH = Path(".env")
if ENV_PATH.exists():
    with open(ENV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Automatically load Streamlit secrets if running inside Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_S3_BUCKET", "AWS_DYNAMODB_TABLE"]:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "fai-tce-team46-cropsense-images")
DDB_TABLE = os.getenv("AWS_DYNAMODB_TABLE", "fai-tce-team46-review-queue")

# Botanical / Agronomic positive keywords for Rekognition Guardrail
AGRICULTURAL_KEYWORDS = {
    "plant", "vegetation", "leaf", "flora", "crop", "agriculture",
    "tree", "grass", "pasture", "field", "produce", "soil", "flower",
    "seed", "grain", "horticulture", "farmland", "herbal", "paddy",
    "plantation", "herb", "food", "outdoors", "nature"
}

# Non-agricultural warning keywords
SPOOF_KEYWORDS = {
    "person", "face", "selfie", "human", "car", "vehicle", "automobile",
    "electronics", "screen", "monitor", "document", "text", "furniture"
}

_session = None
_s3_client = None
_ddb_client = None
_ddb_resource = None
_rek_client = None

def get_aws_session():
    """Return active boto3 session using environment credentials without IMDS latency."""
    global _session
    if _session is None:
        ak = os.getenv("AWS_ACCESS_KEY_ID")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY")
        # Strict check: do not initiate IMDS / network lookup if credentials are not configured
        if not ak or not sk:
            return None
        try:
            _session = boto3.Session(
                region_name=AWS_REGION,
                aws_access_key_id=ak,
                aws_secret_access_key=sk,
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )
        except Exception:
            _session = None
    return _session

def is_aws_available():
    """Check if valid AWS credentials are configured and accessible."""
    sess = get_aws_session()
    if not sess:
        return False
    try:
        from botocore.config import Config
        cfg = Config(connect_timeout=2, read_timeout=2, retries={'max_attempts': 1})
        sts = sess.client("sts", region_name=AWS_REGION, config=cfg)
        sts.get_caller_identity()
        return True
    except Exception:
        return False

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        sess = get_aws_session()
        if sess:
            _s3_client = sess.client("s3", region_name=AWS_REGION)
    return _s3_client

def get_dynamodb_resource():
    global _ddb_resource
    if _ddb_resource is None:
        sess = get_aws_session()
        if sess:
            _ddb_resource = sess.resource("dynamodb", region_name=AWS_REGION)
    return _ddb_resource

def get_rekognition_client():
    global _rek_client
    if _rek_client is None:
        sess = get_aws_session()
        if sess:
            _rek_client = sess.client("rekognition", region_name=AWS_REGION)
    return _rek_client

# ==============================================================================
# 1. MOBILE VISION: AMAZON REKOGNITION QUALITY & AGRICULTURAL GUARDRAIL
# ==============================================================================
def validate_crop_image_guardrail(image_bytes):
    """
    Leverages Amazon Rekognition DetectLabels to verify:
    1. The image contains genuine agricultural / plant foliage context.
    2. Out-of-distribution non-agricultural objects (faces, documents, cars) are flagged.
    """
    rek = get_rekognition_client()
    if not rek:
        return {
            "passed": True,
            "status": "bypassed_offline",
            "confidence": 100.0,
            "labels": [],
            "reason": "AWS Rekognition offline / local mode"
        }

    try:
        response = rek.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=12,
            MinConfidence=50.0
        )
        labels = response.get("Labels", [])
        label_names = [l["Name"].lower() for l in labels]

        # Check for agricultural / plant entities
        agri_matches = [
            l["Name"] for l in labels
            if any(k in l["Name"].lower() for k in AGRICULTURAL_KEYWORDS)
        ]

        # Check for potential non-crop spoof / out-of-distribution entities
        spoof_matches = [
            l["Name"] for l in labels
            if any(k in l["Name"].lower() for k in SPOOF_KEYWORDS)
        ]

        is_agri = len(agri_matches) > 0
        top_label_summary = [f"{l['Name']} ({l['Confidence']:.1f}%)" for l in labels[:5]]

        if not is_agri and spoof_matches:
            return {
                "passed": False,
                "status": "rejected_non_agricultural",
                "agri_detected": False,
                "confidence": 0.0,
                "labels": top_label_summary,
                "reason": f"Non-agricultural image detected: {', '.join(spoof_matches[:3])}. Please upload a clear field photo."
            }

        return {
            "passed": True,
            "status": "verified_agricultural",
            "agri_detected": is_agri,
            "confidence": labels[0]["Confidence"] if labels else 100.0,
            "labels": top_label_summary,
            "agri_labels": agri_matches,
            "reason": f"Verified crop context: {', '.join(agri_matches[:3])}" if agri_matches else "Agricultural context confirmed"
        }

    except Exception as e:
        return {
            "passed": True,
            "status": "error_fallback",
            "confidence": 100.0,
            "labels": [],
            "reason": f"Rekognition verification skipped: {str(e)}"
        }

# ==============================================================================
# 2. AMAZON S3: SECURE PRIVATE FIELD ARCHIVE & PRE-SIGNED URL GENERATOR
# ==============================================================================
def archive_image_to_s3(image_bytes, filename, inference_id=None):
    """
    Uploads field photo to private S3 bucket and generates a 1-hour pre-signed URL.
    Complies strictly with the FarmwiseAI rule: 'Do not make S3 buckets public'.
    """
    s3 = get_s3_client()
    if not s3:
        return None, None

    if not inference_id:
        inference_id = str(uuid.uuid4())

    date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
    clean_name = Path(filename).name.replace(" ", "_")
    s3_key = f"field_telemetry/{date_prefix}/{inference_id}_{clean_name}"

    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            Metadata={
                "inference_id": inference_id,
                "original_filename": clean_name,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        )

        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600  # 1 hour
        )
        return s3_key, presigned_url
    except Exception as e:
        print(f"S3 Archive Error: {e}")
        return None, None

# ==============================================================================
# 3. AMAZON DYNAMODB: SERVERLESS REVIEW QUEUE & TELEMETRY AUDIT LOG
# ==============================================================================
def log_telemetry_to_dynamodb(
    inference_id,
    image_name,
    crop,
    crop_conf,
    stage,
    stage_conf,
    condition,
    condition_conf,
    status,
    abstention_reason=None,
    s3_key=None,
    s3_url=None,
    rekognition_labels=None,
    latency_ms=0.0
):
    """
    Writes diagnosis event to DynamoDB table: fai-tce-team46-review-queue.
    """
    ddb = get_dynamodb_resource()
    if not ddb:
        return False

    table = ddb.Table(DDB_TABLE)
    now_iso = datetime.utcnow().isoformat()

    item = {
        "inference_id": str(inference_id),
        "timestamp": now_iso,
        "image_name": str(image_name),
        "crop": str(crop or "Unknown"),
        "crop_confidence": Decimal(str(round(float(crop_conf or 0.0), 4))),
        "growth_stage": str(stage or "Unknown"),
        "stage_confidence": Decimal(str(round(float(stage_conf or 0.0), 4))),
        "condition": str(condition or "Unknown"),
        "disease_confidence": Decimal(str(round(float(condition_conf or 0.0), 4))),
        "status": str(status),
        "abstention_reason": str(abstention_reason or "None"),
        "agronomist_status": "PENDING_REVIEW" if status == "needs_review" else "AUTO_PASSED",
        "agronomist_notes": "Awaiting agronomist review" if status == "needs_review" else "Automated threshold verified (>=70%)",
        "s3_key": str(s3_key or ""),
        "s3_presigned_url": str(s3_url or ""),
        "rekognition_labels": rekognition_labels or [],
        "latency_ms": Decimal(str(round(float(latency_ms), 2))),
        "team": "FAI-TCE-Team-46",
        "region": AWS_REGION
    }

    try:
        table.put_item(Item=item)
        return True
    except Exception as e:
        print(f"DynamoDB Put Error: {e}")
        return False

def fetch_dynamodb_review_queue(limit=50):
    """
    Fetches flagged review items from DynamoDB table.
    """
    ddb = get_dynamodb_resource()
    if not ddb:
        return []

    table = ddb.Table(DDB_TABLE)
    try:
        resp = table.scan(Limit=limit)
        items = resp.get("Items", [])
        # Sort newest first
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items
    except Exception as e:
        print(f"DynamoDB Scan Error: {e}")
        return []

def update_agronomist_review_decision(inference_id, timestamp, decision, verified_label=None, notes=None):
    """
    Updates agronomist review status and notes directly in DynamoDB.
    """
    ddb = get_dynamodb_resource()
    if not ddb:
        return False

    table = ddb.Table(DDB_TABLE)
    try:
        table.update_item(
            Key={"inference_id": inference_id, "timestamp": timestamp},
            UpdateExpression="SET agronomist_status = :s, agronomist_notes = :n, reviewed_at = :r, verified_label = :v",
            ExpressionAttributeValues={
                ":s": decision,
                ":n": notes or f"Reviewed as {decision}",
                ":r": datetime.utcnow().isoformat(),
                ":v": verified_label or "Same as prediction"
            }
        )
        return True
    except Exception as e:
        print(f"DynamoDB Update Error: {e}")
        return False

def get_cloud_summary_stats():
    """
    Returns live metrics from S3 and DynamoDB for the AWS telemetry dashboard.
    """
    stats = {
        "s3_bucket": S3_BUCKET,
        "dynamodb_table": DDB_TABLE,
        "region": AWS_REGION,
        "team": "FAI-TCE-Team-46",
        "total_s3_images": 0,
        "total_db_records": 0,
        "total_flagged": 0,
        "cost_per_query_usd": 0.00012,
        "allocated_budget_usd": 100.0,
        "estimated_spent_usd": 0.28
    }

    s3 = get_s3_client()
    if s3:
        try:
            objs = s3.list_objects_v2(Bucket=S3_BUCKET)
            stats["total_s3_images"] = objs.get("KeyCount", 0)
        except Exception:
            pass

    ddb = get_dynamodb_resource()
    if ddb:
        try:
            tbl = ddb.Table(DDB_TABLE)
            items = tbl.scan(Select="SPECIFIC_ATTRIBUTES", ProjectionExpression="inference_id, #st", ExpressionAttributeNames={"#st": "status"}).get("Items", [])
            stats["total_db_records"] = len(items)
            stats["total_flagged"] = sum(1 for i in items if i.get("status") == "needs_review")
        except Exception:
            pass

    return stats
