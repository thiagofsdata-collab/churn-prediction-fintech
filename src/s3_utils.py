"""
s3_utils.py
-----------
AWS S3 utilities for model artifact management.

In production ML systems, models are never stored in Git.
They live in versioned object storage (S3, GCS, Azure Blob).
This module provides a clean interface for uploading and
downloading artifacts from S3.
"""

import os
import boto3
import joblib
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError


BUCKET_NAME = "churn-prediction-fintech-artifacts"
S3_PREFIX   = "models/"


def get_s3_client():
    """
    Returns a boto3 S3 client.
    Credentials are loaded from ~/.aws/credentials or environment variables.
    Never hardcode credentials in code.
    """
    try:
        client = boto3.client("s3")
        return client
    except NoCredentialsError:
        raise RuntimeError(
            "AWS credentials not found. "
            "Run 'aws configure' or set environment variables."
        )


def upload_artifact(local_path: str,
                    s3_key: str,
                    bucket: str = BUCKET_NAME) -> bool:
    """
    Uploads a local file to S3.

    Args:
        local_path: Path to the local file
        s3_key: S3 object key (path within bucket)
        bucket: S3 bucket name

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()

    try:
        client.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded: {local_path} → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"Upload failed: {e}")
        return False


def download_artifact(s3_key: str,
                      local_path: str,
                      bucket: str = BUCKET_NAME) -> bool:
    """
    Downloads a file from S3 to a local path.

    Args:
        s3_key: S3 object key
        local_path: Local destination path
        bucket: S3 bucket name

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()

    try:
        client.download_file(bucket, s3_key, local_path)
        print(f"Downloaded: s3://{bucket}/{s3_key} → {local_path}")
        return True
    except ClientError as e:
        print(f"Download failed: {e}")
        return False


def load_model_from_s3(s3_key: str,
                        bucket: str = BUCKET_NAME):
    """
    Downloads a joblib model from S3 and loads it into memory.
    Uses a temp file to avoid leaving artifacts on disk.

    Returns:
        Loaded sklearn Pipeline object
    """
    client = get_s3_client()

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        client.download_file(bucket, s3_key, tmp_path)
        model = joblib.load(tmp_path)
        print(f"Model loaded from s3://{bucket}/{s3_key}")
        return model
    except ClientError as e:
        print(f"Load failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def list_artifacts(prefix: str = S3_PREFIX,
                   bucket: str = BUCKET_NAME) -> list:
    """
    Lists all objects in the S3 bucket under a given prefix.
    """
    client = get_s3_client()

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get("Contents", [])

        if not objects:
            print(f"No artifacts found under s3://{bucket}/{prefix}")
            return []

        print(f"Artifacts in s3://{bucket}/{prefix}:")
        for obj in objects:
            print(f"  {obj['Key']:<50} {obj['Size']:>10} bytes  "
                  f"{obj['LastModified'].strftime('%Y-%m-%d %H:%M')}")
        return objects

    except ClientError as e:
        print(f"List failed: {e}")
        return []