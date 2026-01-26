import boto3
from botocore.client import Config

# Configuration
BUCKET_NAME = 'llm-data'
ENDPOINT = 'https://s3-han02.fptcloud.com'
ACCESS_KEY = 'BO40L9SQNZ7VS944GZ6T' # Replace with your preferred key
SECRET_KEY = 'wS6jOi26RE7vvmWmLxWyYMPpXYcwG8u830Ca2UAQ'      # You must use the secret key paired with the access key
REGION_NAME = 'HN-02'

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=ENDPOINT,
    # Important: Many S3-compatible providers require Signature V4
    config=boto3.session.Config(signature_version='s3v4'),
    verify=False,
    # region_name=REGION_NAME
)

def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"Successfully uploaded {file_name} to {bucket}/{object_name}")
    except Exception as e:
        print(f"Error: {e}")

# Usage
upload_file('test2.txt', BUCKET_NAME)