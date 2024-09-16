import os

import boto3
import botocore
from opensearchpy import OpenSearch
        

def get_opensearch_client(opensearch_host: str, opensearch_port: int) -> OpenSearch:
    """
    Create and return an OpenSearch client.

    This method creates an OpenSearch client with SSL/TLS enabled but hostname
    verification disabled. It uses the OPENSEARCH_PASSWORD environment variable
    for authentication.

    Returns:
        OpenSearch: An initialized OpenSearch client.

    Raises:
        Exception: If the OPENSEARCH_PASSWORD environment variable is not set.

    Note:
        This function uses SSL/TLS without certificate verification, which may
        be insecure in production environments.
    """
    password = os.environ.get('OPENSEARCH_PASSWORD') or ''
    if len(password) == 0:
        raise Exception("OPENSEARCH_PASSWORD environment variable is not set.")
    auth = ('admin', password)

    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    client = OpenSearch(
        hosts=[{"host": opensearch_host, "port": opensearch_port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    info = client.info()
    print(f"Welcome to {info['version']['distribution']} {info['version']['number']}!")
    return client


class Clients:
    def __init__(self, aws_region: str, opensearch_host: str, opensearch_port: int):
        self.opensearch_client = get_opensearch_client(opensearch_host, opensearch_port)
        # AWS SDK for Python (Boto3) clients
        bedrock_config = botocore.config.Config(read_timeout=60, retries={'max_attempts': 3})
        self.bedrock_runtime_client = boto3.client('bedrock-runtime', region_name=aws_region, config=bedrock_config)
        self.iam_client = boto3.client('iam', region_name=aws_region)
        self.lambda_client = boto3.client('lambda', region_name=aws_region)
        self.polly_client = boto3.client('polly', region_name=aws_region)

