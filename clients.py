import os

import boto3
import botocore
from opensearchpy import OpenSearch

from config import Config

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
    def __init__(self, config: Config):
        self.opensearch_client = get_opensearch_client(config.OPENSEARCH_HOST, config.OPENSEARCH_PORT)
        # AWS SDK for Python (Boto3) clients
#       bedrock_config = botocore.config.Config(connect_timeout=300, read_timeout=300, retries={'max_attempts': 3}) # retries={"mode": "standard"}
        bedrock_config = botocore.config.Config(connect_timeout=5, read_timeout=300) 
        self.bedrock_runtime_client_text_model = boto3.client('bedrock-runtime', region_name=config.TEXT_MODEL_REGION, config=bedrock_config)
        self.bedrock_runtime_client_image_model = boto3.client('bedrock-runtime', region_name=config.IMAGE_GENERATION_MODEL_REGION, config=bedrock_config)
        self.bedrock_runtime_client_embedding_multimodal_model = boto3.client('bedrock-runtime', region_name=config.EMBEDDING_MULTIMODAL_MODEL_REGION, config=bedrock_config)
        self.bedrock_runtime_client_embedding_text_model = boto3.client('bedrock-runtime', region_name=config.EMBEDDING_TEXT_MODEL_REGION, config=bedrock_config)
        self.iam_client = boto3.client('iam', region_name=config.TEXT_MODEL_REGION)
        self.lambda_client = boto3.client('lambda', region_name=config.AWS_LAMBDA_FUNCTION_REGION)
        self.polly_client = boto3.client('polly', region_name=config.TEXT_MODEL_REGION)

