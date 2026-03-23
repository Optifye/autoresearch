"""Helpers for loading AWS credentials for the project."""

import boto3

def get_boto3_client(service_name: str):
    """Return a boto3 client with region us-east-1 (no manual credential check)."""
    return boto3.client(service_name, region_name="us-east-1")

def get_s3_client():
    """Shortcut for an S3 client with region us-east-1."""
    return get_boto3_client("s3")
