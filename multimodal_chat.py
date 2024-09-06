#!/usr/bin/env python3

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import re
import time
import urllib
import urllib.request

from datetime import datetime
from html import escape
from urllib.parse import urlparse
from typing import Generator

import boto3

import gradio as gr
from gradio.components.chatbot import FileMessage
from gradio.components.multimodal_textbox import MultimodalData

from rich import print

from selenium import webdriver
import html2text

from duckduckgo_search import DDGS

import wikipedia

from opensearchpy import OpenSearch, NotFoundError
from opensearchpy.helpers import bulk

from PIL import Image

import pypandoc
from pypdf import PdfReader

# Import constants from config_loader
from config_loader import *

# Fix for "Error: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead."
# No other need to import numpy than for this fix
import numpy as np
np.float_ = np.float64

# Fix to avoid the "The current process just got forked..." warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_json_config(filename: str) -> dict:
    """
    Load a JSON configuration file and return its contents as a dictionary.

    Args:
        filename (str): The path to the JSON file to be loaded.

    Returns:
        dict: The contents of the JSON file as a dictionary.

    Raises:
        JSONDecodeError: If the file is not valid JSON.
        FileNotFoundError: If the specified file does not exist.
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


opensearch_client = None

# AWS SDK for Python (Boto3) clients
bedrock_runtime_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
iam_client = boto3.client('iam', region_name=AWS_REGION)
lambda_client = boto3.client('lambda', region_name=AWS_REGION)

TOOLS = load_json_config("./Config/tools.json")
TEXT_INDEX_CONFIG = load_json_config("./Config/text_vector_index.json")
MULTIMODAL_INDEX_CONFIG = load_json_config("./Config/multimodal_vector_index.json")
EXAMPLES = load_json_config('./Config/examples.json')


def add_as_output(content:dict|str, state:dict):
    """
    Add content to the output state, avoiding duplicate images.

    This function checks if the content is an image (has an 'image_id') and ensures
    that duplicate images are not added to the output. If the content is not a
    duplicate image or is not an image at all, it is appended to the output list.

    Args:
        content (dict): The content to be added to the output. May contain an 'image_id' key.
        state (dict): The current state containing the 'output' list.

    Returns:
        None

    Note:
        If a duplicate image is detected, the function returns early without adding it.
    """
    # Avoid to show duplicate images
    if 'image_id' in content:
        image = content
        for item in state['output']:
            if 'image_id' in item and item['image_id'] == image['image_id']:
                return
    state['output'].append(content)


def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch client.

    This function creates an OpenSearch client with SSL/TLS enabled but hostname
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
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    info = client.info()
    print(f"Welcome to {info['version']['distribution']} {info['version']['number']}!")
    return client


def delete_index(client: OpenSearch, index_name: str) -> None:
    """
    Delete an index from OpenSearch if it exists.

    Args:
        client (OpenSearch): The OpenSearch client.
        index_name (str): The name of the index to be deleted.

    Note:
        This function prints the result of the deletion attempt or any exception that occurs.
    """
    if client.indices.exists(index=index_name):
        try:
            _ = client.indices.delete(
                index=index_name,
            )
            print(f"Index {index_name} deleted.")
        except Exception as ex: print(ex)


def create_index(client: OpenSearch, index_name: str, index_config: dict) -> None:
    """
    Create an index in OpenSearch if it doesn't already exist.

    Args:
        client (OpenSearch): The OpenSearch client.
        index_name (str): The name of the index to be created.
        index_config (dict): The configuration for the index.

    Returns:
        None

    Raises:
        Exception: If there's an error during index creation.

    Note:
        This function prints the result of the creation attempt.
        If the index already exists, no action is taken.
    """
    if not client.indices.exists(index=index_name):
        try:
            _ = client.indices.create(
                index=index_name,
                body=index_config,
            )
            print(f"Index {index_name} created.")
        except Exception as ex: print(ex)


def print_index_info(client: OpenSearch, index_name: str) -> None:
    """
    Print information about the multimodal index.

    This function attempts to retrieve and print the configuration of the
    MULTIMODAL_INDEX_NAME index from OpenSearch. If successful, it prints
    the index information in a formatted JSON structure. If an exception
    occurs during the process, it prints the exception details.

    Global variables:
        opensearch_client: The OpenSearch client used to interact with the index.

    Note:
        This function assumes that the opensearch_client has been properly
        initialized and that the MULTIMODAL_INDEX_NAME constant is defined.
    """
    try:
        response = client.indices.get(MULTIMODAL_INDEX_NAME)
        print(json.dumps(response, indent=2))
    except Exception as ex: print(ex)


def get_image_bytes(image_source: str, max_image_size: int | None = None, max_image_dimension: int | None = None) -> bytes:
    """
    Retrieve image bytes from a source and optionally resize the image.

    This function can handle both URL and local file path sources. It will
    resize the image if it exceeds the specified maximum size or dimension.

    Args:
        image_source (str): URL or local path of the image.
        max_image_size (int, optional): Maximum allowed size of the image in bytes.
        max_image_dimension (int, optional): Maximum allowed dimension (width or height) of the image.

    Returns:
        bytes: The image data as bytes, potentially resized.

    Note:
        If resizing is necessary, the function will progressively reduce the image size
        until it meets the specified constraints. The resized image is saved in JPEG format.
    """
    if image_source.startswith(('http://', 'https://')):
        # Download image from URL
        with urllib.request.urlopen(image_source) as response:
            image_bytes = io.BytesIO(response.read())
    else:
        # Open image from local path
        image_bytes = io.BytesIO()
        with open(image_source, 'rb') as f:
            image_bytes.write(f.read())

    original_image_bytes = io.BytesIO(image_bytes.getvalue())
    image_size = len(image_bytes.getvalue())

    divide_by = 1
    while True:
        image_bytes.seek(0)  # Reset the file pointer to the beginning
        with Image.open(original_image_bytes) as img:
            if divide_by > 1:
                resize_comment = f"Divided by {divide_by}"
                img = img.resize(tuple(x // divide_by for x in img.size))
                image_bytes = io.BytesIO()
                img.save(image_bytes, format="JPEG", quality=JPEG_SAVE_QUALITY)
                image_size = image_bytes.tell()
            else:
                resize_comment = "Original"

            print(f"{resize_comment} size {image_size} bytes, dimensions {img.size}")

            if ((max_image_size is None or image_size <= max_image_size) and
                (max_image_dimension is None or all(s <= max_image_dimension for s in img.size))):
                print("Image within required size and dimensions.")
                break

            divide_by *= 2

    return image_bytes.getvalue()


def get_image_base64(image_source: str, max_image_size: int | None = None, max_image_dimension: int | None = None) -> str:
    """
    Convert an image to a base64-encoded string, with optional resizing.

    Args:
        image_source (str): URL or local path of the image.
        max_image_size (int, optional): Maximum allowed size of the image in bytes.
        max_image_dimension (int, optional): Maximum allowed dimension (width or height) of the image.

    Returns:
        str: Base64-encoded string representation of the image.

    Note:
        This function uses get_image_bytes to retrieve and potentially resize the image
        before encoding it to base64.
    """
    image_bytes = get_image_bytes(image_source, max_image_size, max_image_dimension)
    return base64.b64encode(image_bytes).decode('utf-8')


def get_embedding(image_base64: str | None = None, input_text: str | None = None, multimodal: bool = False) -> list[float] | None:
    """
    Generate an embedding vector for the given image and/or text input using Amazon Bedrock.

    This function can handle text-only, image-only, or multimodal (text + image) inputs.
    It selects the appropriate embedding model based on the input types and the multimodal flag.

    Args:
        image_base64 (str | None, optional): Base64-encoded image string. Defaults to None.
        input_text (str | None, optional): Text input for embedding. Defaults to None.
        multimodal (bool, optional): Flag to force multimodal embedding. Defaults to False.

    Returns:
        list[float] | None: The embedding vector as a list of floats, or None if no valid input is provided.

    Raises:
        Exception: If there's an error in the Bedrock API call.

    Note:
        - The function uses global variables EMBEDDING_MULTIMODAL_MODEL_ID and EMBEDDING_TEXT_MODEL_ID
          to determine which model to use.
        - The bedrock_runtime_client is assumed to be a global or imported variable.
    """
    body = {}
    if input_text is not None:
        body["inputText"] = input_text

    if image_base64 is not None:
        body["inputImage"] = image_base64

    if multimodal or 'inputImage' in body:
        embedding_model_id = EMBEDDING_MULTIMODAL_MODEL_ID
    elif 'inputText' in body:
        embedding_model_id = EMBEDDING_TEXT_MODEL_ID
    else:
        return None

    response = bedrock_runtime_client.invoke_model(
        body=json.dumps(body),
        modelId=embedding_model_id,
        accept="application/json", contentType="application/json",
    )

    response_body = json.loads(response.get('body').read())
    finish_reason = response_body.get("message")
    if finish_reason is not None:
        print(finish_reason)
        print(f"Body: {body}")
    embedding_vector = response_body.get("embedding")

    return embedding_vector


def invoke_text_model(messages: list[dict], system_prompt: str | None = None, temperature: float = 0, tools: list[dict] | None = None, return_last_message_only: bool = False) -> dict | str:
    """
    Invoke the text model using Amazon Bedrock's converse API.

    This function prepares the request body, handles retries for throttling exceptions,
    and processes the response from the model.

    Args:
        messages (list): List of message dictionaries to be sent to the model.
        system_prompt (str, optional): System prompt to be added to the request. Defaults to None.
        temperature (float, optional): Temperature setting for the model. Defaults to 0.
        tools (list, optional): List of tools to be used by the model. Defaults to None.
        return_last_message_only (bool, optional): If True, returns only the last message from the model. Defaults to False.

    Returns:
        dict or str: If return_last_message_only is False, returns the full response dictionary.
                     If True, returns only the text of the last message from the model.
                     In case of an error, returns an error message string.

    Raises:
        Exception: Propagates any exceptions not related to throttling.

    Note:
        This function uses global variables MODEL_ID, MAX_TOKENS, MIN_RETRY_WAIT_TIME, and MAX_RETRY_WAIT_TIME.
        It also uses the global bedrock_runtime_client for API calls.
    """
    global bedrock_runtime_client

    converse_body = {
        "modelId": MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": MAX_TOKENS,
            "temperature": temperature,
            },
    }

    if system_prompt is not None:
        converse_body["system"] = [{"text": system_prompt}]

    if tools:
        converse_body["toolConfig"] = {"tools": tools}

    print("Thinking...")

    # To handle throttling retries
    retry_wait_time = MIN_RETRY_WAIT_TIME
    retry_flag = True

    while(retry_flag and retry_wait_time <= MAX_RETRY_WAIT_TIME):
        try:
            response = bedrock_runtime_client.converse(**converse_body)
            retry_flag = False
        except Exception as ex:
            print(ex)
            if ex.response['Error']['Code'] == 'ThrottlingException':
                print(f"Waiting {retry_wait_time} seconds...")
                time.sleep(retry_wait_time)
                # Double the wait time for the next try
                retry_wait_time *= 2
                print("Retrying...")
            else:
                # Handle other client errors
                error_message = f"Error: {ex}"
                print(error_message)
                return error_message

    token_usage = response['usage']
    print(f"Input/Output/Total tokens: {token_usage['inputTokens']}/{token_usage['outputTokens']}/{token_usage['totalTokens']}")
    print(f"Stop reason: {response['stopReason']}")

    if return_last_message_only:
        response_message = response['output']['message']
        last_message = response_message['content'][0]['text']
        return last_message

    return response


def get_base_url(url: str) -> str:
    """
    Extract the base URL from a given URL.

    This function takes a full URL and returns its base URL, which consists
    of the scheme (e.g., 'http', 'https') and the network location (domain).

    Args:
        url (str): The full URL to be processed.

    Returns:
        str: The base URL, in the format "scheme://domain/".

    Example:
        >>> get_base_url("https://www.example.com/page?param=value")
        "https://www.example.com/"
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


def mark_down_formatting(html_text: str, url: str) -> str:
    """
    Convert HTML text to Markdown format with preserved hyperlinks and images.

    This function takes HTML text and a base URL, then converts the HTML to Markdown
    while maintaining the structure of links and images. It uses the html2text library
    for the conversion process.

    Args:
        html_text (str): The HTML text to be converted to Markdown.
        url (str): The base URL used for resolving relative links.

    Returns:
        str: The converted Markdown text.

    Note:
        This function preserves image links, hyperlinks, and list structures.
        It disables line wrapping and converts relative URLs to absolute URLs
        based on the provided base URL.
    """
    h = html2text.HTML2Text()

    base_url = get_base_url(url)

    # Options to transform URL into absolute links
    h.body_width = 0  # Disable line wrapping
    h.ignore_images = False  # Preserve image links
    h.ignore_links = False  # Preserve hyperlinks
    h.wrap_links = False
    h.wrap_list = True
    h.baseurl = base_url

    markdown_text = h.handle(html_text)

    return markdown_text


def remove_specific_xml_tags(text: str) -> str:
    """
    Remove specific XML tags and their content from the input text.

    This function removes specified XML tags and their content completely from the input text.
    It targets specific tags related to quality scores and reflections.

    Args:
        text (str): The input text containing XML tags.

    Returns:
        str: The cleaned text with specified XML tags and their content removed.

    Note:
        This function uses regular expressions for tag removal, which may not be suitable for
        processing very large XML documents due to performance considerations.
    """
    cleaned_text = text

    # Remove specific XML tags and their content
    for tag in ['search_quality_reflection', 'search_quality_score', 'image_quality_score']:
        cleaned_text = re.sub(fr'<{tag}>.*?</{tag}>', '', cleaned_text, flags=re.DOTALL)

    return cleaned_text


def with_xml_tag(text: str, tag: str) -> str:
    """
    Wrap the given text with specified XML tags.

    Args:
        text (str): The text to be wrapped.
        tag (str): The XML tag to use for wrapping.

    Returns:
        str: The text wrapped in the specified XML tags.

    Example:
        >>> with_xml_tag("Hello, World!", "greeting")
        '<greeting>Hello, World!</greeting>'
    """
    return f"<{tag}>{text}</{tag}>"


def split_text_for_collection(text: str) -> list[str]:
    """
    Split the input text into chunks suitable for indexing or processing.

    This function splits the input text into chunks based on sentence boundaries
    and length constraints. It aims to create chunks that are between MIN_CHUNK_LENGTH
    and MAX_CHUNK_LENGTH characters long, while trying to keep sentences together.

    Args:
        text (str): The input text to be split into chunks.

    Returns:
        list: A list of text chunks, where each chunk is a string.

    Note:
        - The function uses regular expressions to split the text into sentences.
        - It attempts to keep sentences together in chunks when possible.
        - The constants MIN_CHUNK_LENGTH and MAX_CHUNK_LENGTH should be defined
          elsewhere in the code to control the size of the chunks.
    """
    chunks = []

    sentences = re.split(r'\. |\n|[)}\]][^a-zA-Z0-9]*[({\[]', text)

    chunk = ''
    next_chunk = ''
    for sentence in sentences:
        sentence = sentence.strip(' \n')
        if len(chunk) < MAX_CHUNK_LENGTH:
            chunk += sentence + "\n"
            if len(chunk) > MIN_CHUNK_LENGTH:
                next_chunk += sentence + "\n"
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
            chunk = next_chunk
            next_chunk = ''

    if len(chunk) > 0:
        chunks.append(chunk)

    return chunks


def add_to_text_index(text: str, id: str, metadata: dict, metadata_delete: dict|None=None) -> None:
    """
    Add text content to the text index in OpenSearch.

    This function processes the input text, splits it into chunks, computes embeddings,
    and indexes the documents in OpenSearch. It can optionally delete existing content
    based on metadata before indexing new content.

    Args:
        text (str): The text content to be indexed.
        id (str): A unique identifier for the text content.
        metadata (dict): Additional metadata to be stored with the text.
        metadata_delete (dict|None, optional): Metadata used to delete existing content
                                               before indexing. Defaults to None.

    Returns:
        None

    Behavior:
        1. If metadata_delete is provided, it deletes existing content matching that metadata.
        2. Splits the input text into chunks.
        3. Processes each chunk in parallel, computing embeddings.
        4. Indexes all processed chunks in bulk to OpenSearch.
        5. Prints information about the indexing process.

    Note:
        This function uses global variables opensearch_client, TEXT_INDEX_NAME, and MAX_WORKERS.
        It also relies on external functions split_text_for_collection and get_embedding.
    """
    global opensearch_client

    if metadata_delete is not None:
        # Delete previous content
        delete_query = {
            "query": {
                "match": metadata_delete
            }
        }
        response = opensearch_client.delete_by_query(
            index=TEXT_INDEX_NAME,
            body=delete_query,
        )
        print(f"Deleted: {response['deleted']}")

    chunks = split_text_for_collection(text)
    print(f"Split into {len(chunks)} chunks")

    def process_chunk(i, chunk, metadata, id):
        formatted_metadata = '\n '.join([f"{key}: {value}" for key, value in metadata.items()])
        chunk = f"{formatted_metadata}\n\n{chunk}"
        print(f"Embedding chunk {i} of {len(chunks)} – Len: {len(chunk)}")
        text_embedding = get_embedding(input_text=chunk)
        document = {
            "id": f"{id}_{i}",
            "document": chunk,
            "embedding_vector": text_embedding,
        }
        document = document | metadata
        return document

    # Compute embeddings
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_chunk, i + 1, chunk, metadata, id) for i, chunk in enumerate(chunks)]

        for future in concurrent.futures.as_completed(futures):
            document = future.result()
            documents.append(document)

    print(f"Indexing {len(documents)} chunks...")

    success, failed = bulk(
        opensearch_client,
        documents,
        index=TEXT_INDEX_NAME,
        raise_on_exception=True
    )
    print(f"Indexed {success} documents successfully, {failed} documents failed.")


def add_to_multimodal_index(image: dict, image_base64: str) -> None:
    """
    Add an image and its metadata to the multimodal index in OpenSearch.

    This function computes an embedding vector for the image and its description,
    then indexes this information along with other image metadata in OpenSearch.

    Args:
        image (dict): A dictionary containing image metadata including:
            - format: The image format (e.g., 'png', 'jpeg')
            - filename: The name of the image file
            - description: A textual description of the image
            - id: A unique identifier for the image
        image_base64 (str): The base64-encoded string representation of the image

    Note:
        This function uses the global opensearch_client to interact with OpenSearch.
        It assumes that MULTIMODAL_INDEX_NAME is defined as a global constant.

    Raises:
        Any exceptions raised by the OpenSearch client or the get_embedding function.

    Returns:
        None. The function prints the indexing result to the console.
    """
    global opensearch_client

    embedding_vector = get_embedding(image_base64=image_base64, input_text=image['description'])
    document = {
        "format": image['format'],
        "filename": image['filename'],
        "description": image['description'],
        "embedding_vector": embedding_vector,
    }
    response = opensearch_client.index(
        index=MULTIMODAL_INDEX_NAME,
        body=document,
        id=image['id'],
        refresh=True,
    )
    print(f"Multimodel index result: {response['result']}")


def get_image_hash(image_bytes: bytes) -> str:
    """
    Compute a SHA-256 hash for the given image bytes.

    This function takes the raw bytes of an image and computes a unique
    hash value using the SHA-256 algorithm. This hash can be used as a
    unique identifier for the image content.

    Args:
        image_bytes (bytes): The raw bytes of the image.

    Returns:
        str: A hexadecimal string representation of the SHA-256 hash.

    Note:
        This function is deterministic, meaning the same image bytes
        will always produce the same hash value.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(image_bytes)
    return hash_obj.hexdigest()


def store_image(image_format: str, image_base64: str, import_image_id:str=''):
    """
    Store an image in the file system and index it in the multimodal database.

    This function takes a base64-encoded image, stores it in the file system,
    generates a description using a text model, and indexes it in the multimodal database.

    Args:
        image_format (str): The format of the image (e.g., 'png', 'jpeg').
        image_base64 (str): The base64-encoded string of the image.
        import_image_id (str, optional): An ID to use for importing an existing image. Defaults to ''.

    Returns:
        dict: A dictionary containing the image metadata if successful, None if there's an ID mismatch.

    Note:
        - If the image already exists in the index, it returns the existing metadata without re-indexing.
        - The function uses global variables IMAGE_PATH and IMAGE_DESCRIPTION_PROMPT.
        - It relies on external functions get_image_hash, get_image_by_id, invoke_text_model, and add_to_multimodal_index.
    """
    image_bytes = base64.b64decode(image_base64)
    image_id = get_image_hash(image_bytes)

    if import_image_id != '' and import_image_id != image_id:
        print(f"Image ID mismatch: {import_image_id} != computed {image_id }")
        return None

    image_filename = IMAGE_PATH + image_id + '.' + image_format
    if not os.path.exists(image_filename):
        with open(image_filename, 'wb') as f:
            f.write(image_bytes)

    image = get_image_by_id(image_id)
    if type(image) is dict:
        print("Image already indexed.")
        return image

    messages = [{
        "role": "user",
        "content": [ { "image": { "format": image_format, "source": { "bytes": image_bytes } }},
                    { "text": IMAGE_DESCRIPTION_PROMPT } ],
    }]

    image_description = invoke_text_model(messages, return_last_message_only=True)
    print(f"Image description: {image_description}")

    image = {
        "id": image_id,
        "format": image_format,
        "filename": image_filename,
        "description": image_description,
    }

    add_to_multimodal_index(image, image_base64)

    return image


def get_image_by_id(image_id: str, return_base64: bool = False) -> dict|str:
    """
    Retrieve image metadata from the multimodal index by its ID.

    This function queries the OpenSearch index to fetch metadata for an image
    with the specified ID. It can optionally return the image data as a base64-encoded string.

    Args:
        image_id (str): The unique identifier of the image to retrieve.
        return_base64 (bool, optional): If True, include the base64-encoded image data
                                        in the returned dictionary. Defaults to False.

    Returns:
        dict: A dictionary containing image metadata if the image is found. The dictionary
              includes keys such as 'format', 'filename', 'description', and 'id'.
              If return_base64 is True, it also includes a 'base64' key with the image data.
        str: An error message if the image is not found or if there's an error during retrieval.

    Raises:
        NotFoundError: If the image with the given ID is not found in the index.
        Exception: For any other errors that occur during the retrieval process.

    Note:
        This function uses the global opensearch_client to interact with OpenSearch
        and assumes that MULTIMODAL_INDEX_NAME is defined as a global constant.
    """
    try:
        response = opensearch_client.get(
            id=image_id,
            index=MULTIMODAL_INDEX_NAME,
            _source_includes=["format", "filename", "description"],
        )
        image = response['_source']
        image['id'] = image_id
        if return_base64:
            image['base64'] = get_image_base64(image['filename'])
        return image
    except NotFoundError:
        return "Not found."
    except Exception as ex:
        error_message = f"Error: {ex}"
        print(error_message)
        return error_message


def search_images(query: str, index_name: str) -> list[dict]|str:
    """
    Search for images in the specified OpenSearch index.

    This function executes a search query against the given OpenSearch index
    and returns the matching image metadata.

    Args:
        query (str): The search query to execute.
        index_name (str): The name of the index to search in.

    Returns:
        list: A list of dictionaries containing image metadata if images are found.
        str: An error message if no images are found or there's an error.

    Raises:
        Exception: If there's an error during the search process.

    Note:
        This function uses the global opensearch_client to perform the search.
    """
    try:
        response = opensearch_client.search(
            body=query,
            index=index_name
        )
        hits = response['hits']['hits']
        print(hits)
        images = []
        if len(hits) == 0:
            return "Not found."
        for h in hits:
            image = h['_source']
            image['id'] = h['_id']
            images.append(image)
        return images
    except Exception as ex:
        error_message = f"Error: {ex}"
        print(error_message)
        return error_message


def get_images_by_description(description: str, max_results: int) -> list[dict]:
    """
    Retrieve images from the multimodal index based on a text description.

    This function performs a two-step process:
    1. It uses a multimodal embedding to find similar images in the index.
    2. It then filters these results using a language model to ensure relevance to the description.

    Args:
        description (str): The text description to match against image descriptions.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of dictionaries containing metadata for matching images.

    Note:
        This function uses global variables MULTIMODAL_INDEX_NAME, IMAGE_FILTER_PROMPT,
        and relies on external functions get_embedding, search_images, with_xml_tag,
        and invoke_text_model.

    Raises:
        Any exceptions raised by the called functions are not explicitly handled here.
    """
    multimodal_text_embedding = get_embedding(input_text=description, multimodal=True)
    query = {
        "size": max_results,
        "query": {
            "knn": {
            "embedding_vector": {
                "vector": multimodal_text_embedding,
                "k": max_results
            }
            }
        },
        "_source": ["format", "filename", "description"]
    }
    images = search_images(query, MULTIMODAL_INDEX_NAME)

    print("Filtering results...")

    prompt = (IMAGE_FILTER_PROMPT + '\n' +
              with_xml_tag(json.dumps(images), 'json_list') + '\n' +
              with_xml_tag(description, 'description'))

    messages = [{
        "role": "user",
        "content": [ { "text": prompt } ],
    }]

    response_message = invoke_text_model(messages, return_last_message_only=True)
    filtered_images = json.loads(response_message)

    print(f"From {len(images)} to {len(filtered_images)} images.")

    return filtered_images

def get_images_by_similarity(image_id: str, max_results: int) -> list[dict]:
    """
    Retrieve images from the multimodal index that are similar to a given image.

    This function finds images in the index that are similar to the image specified by the image_id.
    It uses multimodal embedding to compare the reference image with other images in the catalog.

    Args:
        image_id (str): The unique identifier of the reference image.
        max_results (int): The maximum number of similar images to retrieve.

    Returns:
        list: A list of dictionaries containing metadata for similar images, sorted by similarity.
             Each dictionary includes keys such as 'id', 'format', 'filename', and 'description'.
        str: An error message if the reference image is not found or if there's an error in retrieval.

    Note:
        This function uses global variables MULTIMODAL_INDEX_NAME and relies on external functions
        get_image_by_id, get_embedding, and search_images.

    Raises:
        Any exceptions raised by the called functions are not explicitly handled here.
    """
    image = get_image_by_id(image_id, return_base64=True)
    if type(image) is not dict:
        return "Image not found."
    multimodal_embedding = get_embedding(
        input_text=image['description'],
        image_base64=image['base64']
    )
    # To take care that the first image will be the reference one
    max_results += 1
    query = {
        "size": max_results,
        "query": {
            "knn": {
            "embedding_vector": {
                "vector": multimodal_embedding,
                "k": max_results
            }
            }
        },
        "_source": ["format", "filename", "description"]
    }
    similar_images = search_images(query, MULTIMODAL_INDEX_NAME)
    if type(similar_images) is list and len(similar_images) > 0:
        similar_images.pop(0) # First one is the reference image
    return similar_images


def get_random_images(num: int) -> list[dict]:
    """
    Retrieve a specified number of random images from the image catalog.

    Args:
        num (int): The number of random images to retrieve.

    Returns:
        list: A list of dictionaries containing metadata for the randomly selected images.
              Each dictionary includes keys such as 'id', 'format', 'filename', and 'description'.
        str: An error message if there's an issue retrieving the images.

    Note:
        This function uses the global opensearch_client to query the MULTIMODAL_INDEX_NAME.
        It uses a random score function to ensure randomness in the selection.
    """
    query = {
        "size": num,
        "query": { "function_score": { "random_score": {} } },
        "_source": ["format", "filename", "description"]
    }
    images = search_images(query, MULTIMODAL_INDEX_NAME)
    return images


def get_tool_result_python(tool_input: dict, state: dict) -> str:
    """
    Execute a Python script using AWS Lambda and process the result.

    This function sends a Python script to an AWS Lambda function for execution,
    captures the output, and formats it for display in the chat interface.

    Args:
        tool_input (dict): A dictionary containing the 'script' key with the Python code to execute.
        state (dict): The current state of the chat interface.

    Returns:
        str: The output of the Python script execution, wrapped in XML tags.

    Note:
        - The function uses a global variable AWS_LAMBDA_FUNCTION_NAME for the Lambda function name.
        - It adds the script and its output to the chat interface's state for display.
        - The output is truncated if it exceeds MAX_OUTPUT_LENGTH.
    """
    input_script = tool_input["script"]
    print(f"Script:\n{input_script}")
    start_time = time.time()
    event = {"input_script": input_script}
    print("Invoking Lambda function...")
    output = invoke_lambda_function(AWS_LAMBDA_FUNCTION_NAME, event)
    end_time = time.time()
    elapsed_time = end_time - start_time
    len_output = len(output)
    print(f"Len: {len_output}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if len_output == 0:
        warning_message = "No output printed."
        print(warning_message)
        return warning_message
    if len_output > MAX_OUTPUT_LENGTH:
        output = output[:MAX_OUTPUT_LENGTH] + "\n... (truncated)"
    print(f"Output:\n---\n{output}\n---\nThe user will see the script and its output.")
    add_as_output({"format": "text", "text": f"Python script:\n```python\n{input_script}\n```\n"}, state)
    add_as_output({"format": "text", "text": f"Output:\n```\n{output}\n```\n"}, state)

    return f"{with_xml_tag(output, 'output')}"


def get_tool_result_duckduckgo_text(tool_input: dict, _state: dict) -> str:
    """
    Perform a DuckDuckGo text search and store the results in the archive.

    Args:
        tool_input (dict): A dictionary containing the 'keywords' for the search.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results and a message about archiving.

    Note:
        This function uses the global MAX_SEARCH_RESULTS to limit the number of results.
        It also adds the search results to the text index for future retrieval.
    """
    search_keywords = tool_input["keywords"]
    print(f"Keywords: {search_keywords}")
    try:
        results = DDGS().text(search_keywords, max_results=MAX_SEARCH_RESULTS)
        output = json.dumps(results)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Len: {len(output)}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata_key = search_keywords
    metadata = {"duckduckgo_text": metadata_key, "date": current_date}
    metadata_delete = {"duckduckgo_text": metadata_key}
    archive_item = (
        f"DuckDuckGo Text Search Keywords: {search_keywords}\nResult:\n{output}"
    )
    add_to_text_index(archive_item, metadata_key, metadata, metadata_delete)

    return (
        with_xml_tag(output, "output")
        + "\n\nThis result has been stored in the archive for future use."
    )


def get_tool_result_duckduckgo_news(tool_input: dict, _state: dict) -> str:
    """
    Perform a DuckDuckGo news search and store the results in the archive.

    Args:
        tool_input (dict): A dictionary containing the 'keywords' for the search.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results and a message about archiving.

    Note:
        This function uses the global MAX_SEARCH_RESULTS to limit the number of results.
        It also adds the search results to the text index for future retrieval.
    """
    search_keywords = tool_input["keywords"]
    print(f"Keywords: {search_keywords}")
    try:
        results = DDGS().news(search_keywords, max_results=MAX_SEARCH_RESULTS)
        output = json.dumps(results)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Len: {len(output)}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata_key = search_keywords
    metadata = {"duckduckgo_news": metadata_key, "date": current_date}
    metadata_delete = {"duckduckgo_news": metadata_key}
    archive_item = (
        f"DuckDuckGo News Search Keywords: {search_keywords}\nResult:\n{output}"
    )
    add_to_text_index(archive_item, metadata_key, metadata, metadata_delete)

    return (
        with_xml_tag(output, "output")
        + "\n\nThis result has been stored in the archive for future use."
    )


def get_tool_result_duckduckgo_maps(tool_input: dict, _state: dict) -> str:
    """
    Perform a DuckDuckGo maps search and store the results in the archive.

    Args:
        tool_input (dict): A dictionary containing the 'keywords' and 'place' for the search.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results and a message about archiving.

    Note:
        This function uses the global MAX_SEARCH_RESULTS to limit the number of results.
        It also adds the search results to the text index for future retrieval.
    """
    search_keywords = tool_input["keywords"]
    search_place = tool_input["place"]
    print(f"Keywords: {search_keywords}")
    print(f"Place: {search_place}")
    try:
        results = DDGS().maps(
            search_keywords, search_place, max_results=MAX_SEARCH_RESULTS
        )
        output = json.dumps(results)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Len: {len(output)}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata_key = search_keywords + "\n" + search_place
    metadata = {"duckduckgo_maps": metadata_key, "date": current_date}
    metadata_delete = {"duckduckgo_maps": metadata_key}
    archive_item = f"DuckDuckGo Maps Search Keywords: {search_keywords}\nPlace: {search_place}\nResult:\n{output}"
    add_to_text_index(archive_item, metadata_key, metadata, metadata_delete)

    return (
        with_xml_tag(output, "output")
        + "\n\nThis result has been stored in the archive for future use."
    )


def get_tool_result_wikipedia_search(tool_input: dict, _state: dict) -> str:
    """
    Perform a Wikipedia search and return the results.

    Args:
        tool_input (dict): A dictionary containing the 'query' for the search.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results.

    Note:
        This function uses the Wikipedia API to perform the search and returns
        the results as a JSON string wrapped in XML tags.
    """
    search_query = tool_input["query"]
    print(f"Query: {search_query}")
    try:
        results = wikipedia.search(search_query)
        output = json.dumps(results)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Output: {output}")
    print(f"Len: {len(output)}")
    return with_xml_tag(output, "output")


def get_tool_result_wikipedia_geodata_search(tool_input: dict, _state: dict) -> str:
    """
    Perform a Wikipedia geosearch and return the results.

    Args:
        tool_input (dict): A dictionary containing the search parameters:
            - latitude (float): The latitude of the search center.
            - longitude (float): The longitude of the search center.
            - title (str, optional): The title of a page to search for.
            - radius (int, optional): The search radius in meters.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results as a JSON string.

    Note:
        This function uses the Wikipedia API to perform a geosearch and returns
        the results as a JSON string wrapped in XML tags.
    """
    latitude = tool_input["latitude"]
    longitude = tool_input["longitude"]
    search_title = tool_input.get("title")  # Optional
    radius = tool_input.get("radius")  # Optional
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Title: {search_title}")
    print(f"Radius: {radius}")
    try:
        results = wikipedia.geosearch(
            latitude=latitude, longitude=longitude, title=search_title, radius=radius
        )
        output = json.dumps(results)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Output: {output}")
    print(f"Len: {len(output)}")
    return with_xml_tag(output, "output")


def get_tool_result_wikipedia_page(tool_input: dict, _state: dict) -> str:
    """
    Retrieve and process a Wikipedia page, storing its content in the archive.

    This function fetches a Wikipedia page based on the given title, converts its HTML content
    to Markdown format, and stores it in the text index for future retrieval.

    Args:
        tool_input (dict): A dictionary containing the 'title' key with the Wikipedia page title.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: A message indicating that the page content has been stored in the archive.

    Note:
        This function uses the wikipedia library to fetch page content and the mark_down_formatting
        function to convert HTML to Markdown. It also uses add_to_text_index to store the content
        in the archive with appropriate metadata.
    """
    search_title = tool_input["title"]
    print(f"Title: {search_title}")
    try:
        page = wikipedia.page(title=search_title, auto_suggest=False)
        output = mark_down_formatting(page.html(), page.url)
    except Exception as e:
        output = str(e)
    output = output.strip()
    print(f"Len: {len(output)}")
    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata = {"wikipedia_page": search_title, "date": current_date}
    metadata_delete = {"wikipedia_page": search_title}
    add_to_text_index(output, search_title, metadata, metadata_delete)

    return f"The full content of the page has been stored in the archive so that you can retrieve what you need."


def get_tool_result_browser(tool_input: dict, _state: dict) -> str:
    """
    Retrieve and process content from a given URL using Selenium.

    This function uses Selenium WebDriver to navigate to the specified URL,
    retrieve the page content, convert it to Markdown format, and store it
    in the text index for future retrieval.

    Args:
        tool_input (dict): A dictionary containing the 'url' key with the target URL.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: A message indicating that the content has been stored in the archive.

    Note:
        This function uses Selenium with Chrome in headless mode to retrieve page content.
        It also uses mark_down_formatting to convert HTML to Markdown and add_to_text_index
        to store the content in the archive with appropriate metadata.
    """
    url = tool_input["url"]
    print(f"URL: {url}")

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as f:
            status = f.status
            content_length = f.headers.get("Content-Length", 0)
        print(f"Status: {status}")
        print(f"Content length: {content_length}")
    except Exception:
        # Ignore the error and try again with Selenium
        pass

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--incognito")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)

    driver.get(url)

    page = driver.page_source
    print(f"Page length:", len(page))

    markdown_text = mark_down_formatting(page, url)
    print(f"Markdown text length:", len(markdown_text))

    driver.quit()

    if len(markdown_text) < 10:
        return "I am not able or allowed to get content from this URL."

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata = {"url": url, "hostname": hostname, "date": current_date}
    metadata_delete = {"url": url}  # To delete previous content from the same URL
    add_to_text_index(markdown_text, url, metadata, metadata_delete)

    return f"The full content of the URL ({len(markdown_text)} characters) has been stored in the archive. You can retrieve what you need using keywords."


def get_tool_result_retrive_from_archive(tool_input: dict, _state: dict) -> str:
    """
    Retrieve content from the archive based on given keywords.

    This function searches the text index using the provided keywords and returns
    the matching documents.

    Args:
        tool_input (dict): A dictionary containing the 'keywords' to search for.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: XML-tagged output containing the search results as a JSON string.

    Note:
        This function uses global variables TEXT_INDEX_NAME and MAX_ARCHIVE_RESULTS
        to perform the search. It also relies on external functions get_embedding
        and opensearch_client for searching the index.
    """
    keywords = tool_input["keywords"]
    print(f"Keywords: {keywords}")

    text_embedding = get_embedding(input_text=keywords)

    query = {
        "size": MAX_ARCHIVE_RESULTS,
        "query": {"knn": {"embedding_vector": {"vector": text_embedding, "k": 5}}},
        "_source": ["code", "url", "website", "document"],
    }

    try:
        response = opensearch_client.search(body=query, index=TEXT_INDEX_NAME)
    except Exception as ex:
        print(ex)

    documents = ""
    for value in response["hits"]["hits"]:
        documents += json.dumps(value["_source"])

    print(f"Len: {len(documents)}")
    return with_xml_tag(documents, "archive")


def get_tool_result_store_in_archive(tool_input: dict, _state: dict) -> str:
    """
    Store content in the archive.

    This function takes the provided content and stores it in the text index
    with the current date as metadata.

    Args:
        tool_input (dict): A dictionary containing the 'content' key with the text to be stored.
        _state (dict): The current state of the chat interface (unused in this function).

    Returns:
        str: A message indicating whether the content was successfully stored or an error occurred.

    Note:
        This function uses the global function add_to_text_index to store the content in the archive.
    """
    content = tool_input["content"]
    if len(content) == 0:
        return "You need to provide content to store in the archive."
    else:
        print(f"Content:\n---\n{content}\n---")

    current_date = datetime.now().strftime("%Y-%m-%d")
    metadata = {"date": current_date}
    add_to_text_index(content, content, metadata)

    return "The content has been stored in the archive."


def render_notebook(notebook: list[str]) -> str:
    """
    Render a notebook as a single string.

    This function takes a list of strings representing notebook pages and combines them
    into a single string, with each page separated by double newlines. It also removes
    any instances of three or more consecutive newlines, replacing them with double newlines.

    Args:
        notebook (list[str]): A list of strings, each representing a page in the notebook.

    Returns:
        str: A single string containing all notebook pages, properly formatted.
    """
    rendered_notebook = "\n\n".join(notebook)
    rendered_notebook = re.sub(r'\n{3,}', '\n\n', rendered_notebook)
    return rendered_notebook

def get_tool_result_notebook(tool_input: dict, state: dict) -> str:
    """
    Process a notebook command and update the notebook state accordingly.

    This function handles various notebook operations such as starting a new notebook,
    adding pages, reviewing pages, updating pages, and sharing the notebook.

    Args:
        tool_input (dict): A dictionary containing the command and optional content.
        state (dict): The current state of the notebook.

    Returns:
        str: A message indicating the result of the operation.

    Commands:
        - start_new: Initializes a new empty notebook.
        - add_page: Adds a new page to the notebook.
        - start_review: Begins a review of the notebook from the first page.
        - next_page: Moves to the next page during review.
        - update_page: Updates the content of the current page.
        - share_notebook: Shares the entire notebook content.
        - save_notebook_file: Saves the notebook to a file.
        - info: Provides information about the notebook and current page.

    Note:
        This function modifies the 'state' dictionary to keep track of the notebook's
        current state, including the current page and total number of pages.
    """
    command = tool_input.get("command")
    content = tool_input.get("content", "")
    print(f"Command: {command}")
    if len(content) > 0:
        print(f"Content:\n---\n{content}\n---")

    num_pages = len(state["notebook"])

    match command:
        case "start_new":
            state["notebook"] = []
            state["notebook_current_page"] = 0
            return "This is a new notebook. There are no pages. Start by adding some content."
        case "add_page":
            if len(content) == 0:
                return "You need to provide content to add a new page."
            state["notebook"].append(content)
            num_pages = len(state["notebook"])
            state["notebook_current_page"] = num_pages - 1
            return f"New page added at the end. You're now at page {state['notebook_current_page'] + 1} of {num_pages}. You can add more pages, start a review, or share the notebook with the user."
        case "start_review":
            if num_pages == 0:
                return "The notebook is empty. There are no pages to review. Start by adding some content."
            state["notebook_current_page"] = 0
            page_content = state["notebook"][0]
            page_content_with_tag = with_xml_tag(page_content, "page")
            return f"You're starting your review at page 1 of {num_pages}. This is the content of the current page:\n\n{page_content_with_tag}\n\nYou can update the content of this page or move to the next page. The review is completed when you reach the end."
        case "next_page":
            if state["notebook_current_page"] >= num_pages - 1:
                return f"You're at the end. You're at page {state['notebook_current_page'] + 1} of {num_pages}. You can start a review or share the notebook with the user."
            state["notebook_current_page"] += 1
            page_content = state["notebook"][state["notebook_current_page"]]
            page_content_with_tag = with_xml_tag(page_content, "page")
            return f"Moving to the next page. You're now at page {state['notebook_current_page'] + 1} of {num_pages}. This is the content of the current page:\n\n{page_content_with_tag}\n\nYou can update the content of this page or move to the next page. The review is completed when you reach the end."
        case "update_page":
            if num_pages == 0:
                return "The notebook is empty. There are no pages. Start by adding some content."
            if len(content) == 0:
                return "You need to provide content to update the current page."
            state["notebook"][state["notebook_current_page"]] = content
            return f"The current page has been updated with the new content."
        case "share_notebook":
            if num_pages == 0:
                return "The notebook is empty. There are no pages to share."
            print("Sharing the notebook...")
            notebook_output = render_notebook(state["notebook"])
            add_as_output(
                {"format": "text", "text": "This is the content of the notebook:"},
                state,
            )
            add_as_output({"format": "text", "text": "<hr>"}, state)
            add_as_output({"format": "text", "text": notebook_output}, state)
            add_as_output({"format": "text", "text": "<hr>"}, state)
            return f"The notebook ({num_pages} pages) has been shared with the user."
        case "save_notebook_file":
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            notebook_filename = f"{OUTPUT_PATH}notebook_{current_datetime}.md"
            notebook_output = render_notebook(state["notebook"])
            with open(notebook_filename, "w") as file:
                file.write(notebook_output)
            return f"The notebook ({num_pages} pages) has been saved to {notebook_filename}."
        case "info":
            tmp_output = "\n".join(state["notebook"])
            num_chars = len(tmp_output)
            num_words = len(tmp_output.split())
            page_content = state["notebook"][state["notebook_current_page"]]
            num_chars_page = len(page_content)
            num_words_page = len(page_content.split())
            info_message = f"The notebook contains {num_pages} page(s) containing {num_chars} characters or {num_words} words.\nThe current page contains {num_chars_page} characters or {num_words_page} words."
            print(info_message)
            return info_message
        case _:
            return "Invalid command."


def get_tool_result_generate_image(tool_input: dict, state: dict) -> str:
    """
    Generate an image based on a given prompt using Amazon Bedrock's image generation model.

    This function takes a text prompt, prepares the request body for the image generation API,
    invokes the model, and processes the response. If successful, it stores the generated image
    in the image catalog and adds it to the output state for display.

    Args:
        tool_input (dict): A dictionary containing the 'prompt' key with the text description for image generation.
        state (dict): The current state of the chat interface, used for storing output.

    Returns:
        str: A message describing the generated image, including its ID and description.
             If an error occurs, it returns an error message instead.

    Note:
        This function uses global variables for model configuration and client access.
        It also relies on external functions for image storage and output handling.
    """
    prompt = tool_input["prompt"]
    print(f"Prompt: {prompt}")

    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": DEFAULT_IMAGE_HEIGHT,
                "width": DEFAULT_IMAGE_WIDTH,
            },
        }
    )

    try:
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=IMAGE_GENERATION_MODEL_UD,
        )
    except Exception as ex:
        error_message = f"Image generation error. Error is {ex}"
        print(error_message)
        return error_message

    response_body = json.loads(response.get("body").read())
    image_base64 = response_body.get("images")[0]
    image_format = "png"

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        error_message = f"Image generation error. Error is {finish_reason}"
        print(error_message)
        return error_message

    print(f"Image base64 size: {len(image_base64)}")
    image = store_image(image_format, image_base64)
    add_as_output(image, state)

    return f"A new image with with 'image_id' {image['id']} and this description has been stored in the image catalog:\n\n{image['description']}\n\nThe image has been shown to the user.\nDon't mention the 'image_id' in your response."


def get_tool_result_search_image_catalog(tool_input: dict, state: dict) -> str:
    """
    Search for images in the image catalog based on a text description.

    This function retrieves images from the catalog that match a given description,
    adds them to the output state for display, and prepares a summary of the results.

    Args:
        tool_input (dict): A dictionary containing:
            - description (str): The text description to search for.
            - max_results (int, optional): Maximum number of results to return.
              Defaults to MAX_IMAGE_SEARCH_RESULTS.
        state (dict): The current state of the chat interface, used for storing output.

    Returns:
        str: A summary of the search results, including descriptions of found images.
             If no images are found, returns a message indicating so.

    Note:
        This function modifies the 'state' dictionary by adding found images to the output.
        It uses external functions get_images_by_description and add_as_output.
    """
    description = tool_input.get("description")
    max_results = tool_input.get("max_results", MAX_IMAGE_SEARCH_RESULTS)
    print(f"Description: {description}")
    print(f"Max results: {max_results}")
    images = get_images_by_description(description, max_results)
    if type(images) is not list:
        return images  # It's an error
    result = ""
    for image in images:
        add_as_output(image, state)
        result += f"The image with 'image_id' {image['id']} and this description has been shown to the user:\n\n{image['description']}\n\n"
    if len(result) == 0:
        return f"No images found."
    result = f"These are images similar to the description in descreasing order of similarity:\n{result}"
    return result


def get_tool_result_similarity_image_catalog(tool_input: dict, state: dict) -> str:
    """
    Search for similar images in the image catalog based on a reference image.

    This function retrieves images from the catalog that are similar to a given reference image,
    adds them to the output state for display, and prepares a summary of the results.

    Args:
        tool_input (dict): A dictionary containing:
            - image_id (str): The ID of the reference image to search for similar images.
            - max_results (int, optional): Maximum number of results to return.
              Defaults to MAX_IMAGE_SEARCH_RESULTS.
        state (dict): The current state of the chat interface, used for storing output.

    Returns:
        str: A summary of the search results, including descriptions of found images.
             If no similar images are found, returns a message indicating so.
             If an error occurs, returns an error message.

    Note:
        This function modifies the 'state' dictionary by adding found images to the output.
        It uses external functions get_images_by_similarity and add_as_output.
    """
    image_id = tool_input.get("image_id")
    max_results = tool_input.get("max_results", MAX_IMAGE_SEARCH_RESULTS)
    print(f"Image ID: {image_id}")
    print(f"Max results: {max_results}")
    similar_images = get_images_by_similarity(image_id, max_results)
    if type(similar_images) is not list:
        return similar_images  # It's an error
    result = ""
    for image in similar_images:
        assert image_id != image["id"]
        add_as_output(image, state)
        result += f"The image with 'image_id' {image['id']} and this description has been shown to the user:\n\n{image['description']}\n\n"
    if len(result) == 0:
        return f"No similar images found."
    result = f"These are images similar to the reference image in descreasing order of similarity:\n{result}"
    return result


def get_tool_result_random_images(tool_input: dict, state: dict) -> str:
    """
    Retrieve random images from the image catalog and add them to the output state.

    This function fetches a specified number of random images from the image catalog,
    adds them to the output state for display, and prepares a summary of the results.

    Args:
        tool_input (dict): A dictionary containing:
            - num (int): The number of random images to retrieve.
        state (dict): The current state of the chat interface, used for storing output.

    Returns:
        str: A summary of the random images retrieved, including descriptions of each image.
             If no images are returned or an error occurs, returns an appropriate message.

    Note:
        This function modifies the 'state' dictionary by adding retrieved images to the output.
        It uses external functions get_random_images and add_as_output.
    """
    num = tool_input.get("num")
    print(f"Num: {num}")
    random_images = get_random_images(num)
    if type(random_images) is not list:
        return random_images  # It's an error
    result = ""
    for image in random_images:
        add_as_output(image, state)
        result += f"The image with 'image_id' {image['id']} and this description has been shown to the user:\n\n{image['description']}\n\n"
    if len(result) == 0:
        return f"No random images returned."
    result = f"These are random images from the image catalog:\n{result}"
    return result


def get_tool_result_image_catalog_count(_tool_input: dict, _state: dict) -> int | str:
    """
    Count the number of documents in the image catalog.

    This function queries the OpenSearch index to get the total count of images
    in the multimodal index.

    Returns:
        int: The number of images in the catalog.
        str: An error message if an exception occurs during the count operation.

    Note:
        This function uses the global opensearch_client to interact with OpenSearch.
        It assumes that MULTIMODAL_INDEX_NAME is defined as a global constant.
    """
    try:
        info = opensearch_client.count(index=MULTIMODAL_INDEX_NAME)
        print(f"Image catalog info: {info}")
        count = info["count"]
        return count
    except Exception as ex:
        error_message = f"Error: {ex}"
        print(error_message)
        return error_message


def get_tool_result_download_image_into_catalog(tool_input: dict, state: dict) -> str:
    """
    Download an image from a given URL and add it to the image catalog.

    This function retrieves an image from a specified URL, processes it, and stores
    it in the image catalog. It performs the following steps:
    1. Validates the URL and checks the image format.
    2. Downloads the image and converts it to base64.
    3. Stores the image in the catalog using the store_image function.
    4. Adds the image to the output state for display.

    Args:
        tool_input (dict): A dictionary containing the 'url' key with the image URL.
        state (dict): The current state of the chat interface, used for storing output.

    Returns:
        str: A message describing the result of the operation, including the image ID
             and description if successful, or an error message if the operation fails.

    Raises:
        Various exceptions may be raised and caught within the function, resulting
        in error messages being returned instead of the function terminating.

    Note:
        This function uses several global variables and external functions for
        image processing and storage.
    """
    url = tool_input.get("url")
    print(f"URL: {url}")
    if url is None:
        return "You need to provide a URL."

    # Get the image format from the content type, do an http HEAD to the url using urllib
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as f:
            status = f.status
            content_type = f.headers.get("Content-Type", "")
    except Exception as ex:
        error_message = f"Error downloading image: {ex}"
        print(error_message)
        return error_message

    if status >= 400:
        error_message = f"Error downloading image. Status code: {status}"
        print(error_message)
        return error_message

    # Content type - remove everything from ';' onwards (';' included)
    content_type = content_type.split(';')[0]
    type, format = content_type.split('/')

    if type != 'image' or format not in IMAGE_FORMATS:
        error_message = f"Unsupported image format: {content_type}"
        print(error_message)
        return error_message

    # Get file extension from URL
    try:
        image_base64 = get_image_base64(url,
                                        max_image_size=MAX_CHAT_IMAGE_SIZE,
                                        max_image_dimension=MAX_CHAT_IMAGE_DIMENSIONS)
    except Exception as ex:
        error_message = f"Error downloading image: {ex}"
        print(error_message)
        return error_message

    # Convert image bytes to base64
    image = store_image(format, image_base64)
    add_as_output(image, state)

    return f"Image downloaded and stored in the image catalog with 'image_id' {image['id']} and description:\n\n{image['description']}"


def get_tool_result_personal_improvement(tool_input: dict, state: dict) -> str:
    """
    Handle personal improvement commands and update the improvement state.

    This function processes commands related to personal improvements, including
    showing current improvements and updating them.

    Args:
        tool_input (dict): A dictionary containing command information:
            - command (str): The action to perform ('show_improvements' or 'update_improvements').
            - improvements (str, optional): New improvements to be stored, replacing the current ones.
        state (dict): The current state of the chat interface, containing the improvements.

    Returns:
        str: A message indicating the result of the operation.

    Note:
        The function initializes the 'improvements' key in the state if it doesn't exist.
    """
    command = tool_input.get("command")
    improvements = tool_input.get("improvements")
    print(f"Command: {command}")
    print(f"Improvements: {improvements}")
    match command:
        case 'show_improvements':
            return f"These are the current improvements:\n{state['improvements']}"
        case 'update_improvements':
            state['improvements']= improvements
            return"Improvement updated."
        case _:
            return "Invalid command."


class ToolError(Exception):
    """
    Custom exception class for tool-related errors.

    This exception is raised when there's an issue with tool execution or usage
    in the chat system. It can be used to handle specific errors related to
    tools and provide meaningful error messages to the user or the system.
    """
    pass


TOOL_FUNCTIONS = {
    'python': get_tool_result_python,
    'duckduckgo_text': get_tool_result_duckduckgo_text,
    'duckduckgo_news': get_tool_result_duckduckgo_news,
    'duckduckgo_maps': get_tool_result_duckduckgo_maps,
    'wikipedia_search': get_tool_result_wikipedia_search,
    'wikipedia_geodata_search': get_tool_result_wikipedia_geodata_search,
    'wikipedia_page': get_tool_result_wikipedia_page,
    'browser': get_tool_result_browser,
    'retrive_from_archive': get_tool_result_retrive_from_archive,
    'store_in_archive': get_tool_result_store_in_archive,
    'notebook': get_tool_result_notebook,
    'generate_image': get_tool_result_generate_image,
    'search_image_catalog': get_tool_result_search_image_catalog,
    'similarity_image_catalog': get_tool_result_similarity_image_catalog,
    'random_images': get_tool_result_random_images,
    'image_catalog_count': get_tool_result_image_catalog_count,
    'download_image_into_catalog': get_tool_result_download_image_into_catalog,
    'personal_improvement': get_tool_result_personal_improvement,
}


def check_tools_consistency() -> None:
    """
    Check the consistency between defined tools and their corresponding functions.

    This function compares the set of tool names defined in the TOOLS global variable
    with the set of function names in the TOOL_FUNCTIONS dictionary. It ensures that
    there is a one-to-one correspondence between the defined tools and their
    implementation functions.

    Raises:
        Exception: If there is a mismatch between the tools defined in TOOLS
                   and the functions defined in TOOL_FUNCTIONS.

    Note:
        This function assumes that TOOLS and TOOL_FUNCTIONS are global variables
        defined elsewhere in the code.
    """
    tools_set = set([ t['toolSpec']['name'] for t in TOOLS])
    tool_functions_set = set(TOOL_FUNCTIONS.keys())

    if tools_set != tool_functions_set:
        raise Exception(f"Tools and tool functions are not consistent: {tools_set} != {tool_functions_set}")


def get_tool_result(tool_use_block: dict, state: dict) -> str:
    """
    Execute a tool and return its result.

    This function takes a tool use block and the current state, executes the
    specified tool, and returns the result. It handles tool execution errors
    by raising a ToolError.

    Args:
        tool_use_block (dict): A dictionary containing the tool use information,
                               including the tool name and input.
        state (dict): The current state of the application.

    Returns:
        The result of the tool execution.

    Raises:
        ToolError: If an invalid tool name is provided.
    """
    global opensearch_client

    tool_use_name = tool_use_block['name']

    print(f"Using tool {tool_use_name}")

    try:
        return TOOL_FUNCTIONS[tool_use_name](tool_use_block['input'], state)
    except KeyError:
        raise ToolError(f"Invalid function name: {tool_use_name}")


def process_response_message(response_message: dict, state: dict) -> None:
    """
    Process the response message and update the state with the output content.

    Args:
        response_message (dict): The response message from the AI model.
        state (dict): The current state of the application.
    """
    output_content = []
    if response_message['role'] == 'assistant' and 'content' in response_message:
        for c in response_message['content']:
            if 'text' in c:
                output_content.append(c['text'])
            if 'toolUse' in c:
                hidden_content = "<!--\n" + str(c['toolUse']) + "\n-->"
                output_content.append(hidden_content)
    if response_message['role'] == 'user' and 'content' in response_message:
        for c in response_message['content']:
            if 'toolResult' in c:
                tool_result = ''
                content = c['toolResult']['content'][0]
                if 'json' in content:
                    tool_result = json.dumps(content['json'])
                elif 'text' in content:
                    tool_result = content['text']
                if len(tool_result) > 0:
                    # Hidden to not show up in the chat but still be used by the model
                    hidden_content = "<!--\n" + tool_result + "\n-->"
                    output_content.append(hidden_content)
    for m in output_content:
        m = remove_specific_xml_tags(m)
        add_as_output({"format": "text", "text": m}, state)


def handle_response(response_message: dict, state: dict) -> dict|None:
    """
    Handle the response from the AI model and process any tool use requests.

    This function takes the response message from the AI model and the current state,
    processes any tool use requests within the response, and generates follow-up
    content blocks with tool results or error messages.

    Args:
        response_message (dict): The response message from the AI model containing
                                 content blocks and potential tool use requests.
        state (dict): The current state of the chat interface.

    Returns:
        dict or None: A follow-up message containing tool results if any tools were used,
                      or None if no tools were used.

    Note:
        This function handles tool execution errors by catching ToolError exceptions
        and including error messages in the follow-up content blocks.
    """

    response_content_blocks = response_message['content']
    follow_up_content_blocks = []

    for content_block in response_content_blocks:
        if 'toolUse' in content_block:
            tool_use_block = content_block['toolUse']

            try:
                tool_result_value = get_tool_result(tool_use_block, state)

                if tool_result_value is not None:
                    follow_up_content_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_block['toolUseId'],
                            "content": [
                                { "json": { "result": tool_result_value } }
                            ]
                        }
                    })

            except ToolError as e:
                follow_up_content_blocks.append({
                    "toolResult": {
                        "toolUseId": tool_use_block['toolUseId'],
                        "content": [ { "text": repr(e) } ],
                        "status": "error"
                    }
                })

    if len(follow_up_content_blocks) > 0:
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
        return follow_up_message
    else:
        return None


def get_file_name_and_extension(full_file_name: str) -> tuple[str, str]:
    """
    Extract the file name and extension from a full file path.

    This function takes a full file path and returns the file name without the extension
    and the extension separately. The extension is returned in lowercase without the leading dot.

    Args:
        full_file_name (str): The full path of the file including the file name and extension.

    Returns:
        tuple: A tuple containing two elements:
            - file_name (str): The name of the file without the extension.
            - extension (str): The file extension in lowercase without the leading dot.
                               If there's no extension, an empty string is returned.

    Example:
        >>> get_file_name_and_extension('/path/to/myfile.txt')
        ('myfile', 'txt')
        >>> get_file_name_and_extension('document.PDF')
        ('document', 'pdf')
        >>> get_file_name_and_extension('image')
        ('image', '')
    """
    file_name, extension = os.path.splitext(os.path.basename(full_file_name))
    if len(extension) > 0:
        extension = extension[1:].lower() # Remove the leading '.' and make it lowercase
    return file_name, extension


def format_messages_for_bedrock_converse(message: dict, history: list[dict], state: dict) -> list[dict]:
    """
    Format messages for the Bedrock converse API.

    This function takes a message, conversation history, and state, and formats them
    into a structure suitable for the Bedrock Converse API. It processes text and file
    contents, handles different message types, and prepares image and document data.

    Args:
        message (dict): The latest user message.
        history (list): A list of previous messages in the conversation.
        state (dict): The current state of the application.

    Returns:
        list: A list of formatted messages ready for the Bedrock converse API.

    Note:
        This function handles various types of content including text, images, and documents.
        It also processes file uploads and stores images in the catalog when necessary.
    """

    # Temporarily add the latest user message for convinience
    history.append({"role": "user", "content": message})

    messages = []
    message_content = []

    skip_next = False
    for m in history:
        append_message = False
        m_role = m['role']
        m_content = m['content']

        # To skip automatic replies to user empty messages
        if skip_next:
            skip_next = False
            continue
        # To skip user empty messages
        if m_content == '':
            skip_next = True
            continue

        if type(m_content) is MultimodalData:
            m_text = m_content.text
            m_files = [ {"path": file_data.path} for file_data in m_content.files ]
        elif type(m_content) is FileMessage:
            m_text = ""
            m_files = [{"path": m_content.file.path}]
        elif type(m_content) is tuple: # To handle a Gradio bug when content is "(file,)"
            m_text = ""
            m_files = [{"path": m_content[0]}]
        elif type(m_content) is str:
            m_text = m_content
            m_files = []
        else:
            m_text = m_content.get("text", "")
            m_files = m_content.get("files", [])
        if len(m_text) > 0:
            m_text = remove_specific_xml_tags(m_text) # To remove <img> tags
            message_content.append({"text": m_text})
            append_message = True
        for file in m_files:
            file = file['path']
            file_name, extension = get_file_name_and_extension(os.path.basename(file))
            if extension == 'jpg':
                extension = 'jpeg' # Fix
            # png | jpeg | gif | webp
            if extension in IMAGE_FORMATS:

                file_content = get_image_bytes(
                    file,
                    max_image_size=MAX_INFERENCE_IMAGE_SIZE,
                    max_image_dimension=MAX_INFERENCE_IMAGE_DIMENSIONS,
                )
                message_content.append({
                    "image": {
                        "format": extension,
                        "source": {
                            "bytes": file_content
                        }
                    }
                })

                image_base64 = get_image_base64(
                    file,
                    max_image_size=MAX_CHAT_IMAGE_SIZE,
                    max_image_dimension=MAX_CHAT_IMAGE_DIMENSIONS
                )
                image = store_image(extension, image_base64)
                add_as_output(image, state)
                message_content.append({
                    "text": f"The previous image has been stored in the image catalog with 'image_id': {image['id']}"
                })

                append_message = True
            # pdf | csv | doc | docx | xls | xlsx | html | txt | md
            elif HANDLE_DOCUMENT_TO_TEXT_IN_CODE:
                print(f"Importing '{file_name}.{extension}'...")
                try:
                    if extension == 'pdf':
                        text_pages = []
                        reader = PdfReader(file)
                        for page in reader.pages:
                            text = page.extract_text()
                            text_pages.append(with_xml_tag(text, 'page'))
                        file_text = '\n'.join(text_pages)
                    else:
                        file_text = pypandoc.convert_file(file, 'rst')
                    file_message = f"This is the text content of the '{file_name}.{extension}' file:\n\n" + with_xml_tag(file_text, "file")
                    message_content.append({ "text": file_message })
                except Exception as ex:
                    error_message = f"Error processing {file_name}.{extension} file: {ex}"
                    print(error_message)
                    message_content.append({ "text": error_message })
            elif extension in DOCUMENT_FORMATS:
                with open(file, 'rb') as f:
                    file_content = f.read()
                message_content.append({
                    "document": {
                        "name": file_name,
                        "format": extension,
                        "source": {
                            "bytes": file_content
                        }
                    }
                })
            else:
                print(f"Unsupported file type: {extension}")
        if append_message:
            if len(messages) > 0 and m_role == messages[-1]['role']:
                messages[-1]['content'].extend(message_content)
            else:
                messages.append({"role": m_role, "content": message_content})
            message_content = []

    # Remove the last user message
    history.pop()

    return messages


def manage_conversation_flow(messages: list[dict], system_prompt: str, temperature: float, state: dict) -> None:
    """
    Run a conversation loop with the AI model, processing responses and handling tool usage.

    This function manages the conversation flow between the user and the AI model. It sends messages
    to the model, processes the responses, handles any tool usage requests, and continues the
    conversation until a final response is ready or the maximum number of loops is reached.

    Args:
        messages (list): A list of message dictionaries representing the conversation history.
        system_prompt (str): The system prompt to guide the AI model's behavior.
        temperature (float): The temperature parameter for the AI model's response generation.
        state (dict): The current state of the chat interface.

    Returns:
        str: The final response from the AI model, with XML tags removed.

    Note:
        This function uses global variables MAX_LOOPS and TOOLS, and relies on external functions
        invoke_text_model, handle_response, and remove_xml_tags.
    """
    loop_count = 0
    continue_loop = True

    # Add current date and time to the system prompt
    current_date_and_day_of_the_week = datetime.now().strftime("%a, %Y-%m-%d")
    current_time = datetime.now().strftime("%I:%M:%S %p")
    system_prompt_with_improvements = system_prompt + f"\nKeep in mind that today is {current_date_and_day_of_the_week} and the current time is {current_time}."

    if len(state['improvements']) > 0:
        system_prompt_with_improvements += "\n\nImprovements:\n" + state['improvements']

    while continue_loop:

        response = invoke_text_model(messages, system_prompt_with_improvements, temperature, tools=TOOLS)

        response_message = response['output']['message']
        messages.append(response_message)

        process_response_message(response_message, state)

        loop_count = loop_count + 1

        if loop_count >= MAX_LOOPS:
            print(f"Hit loop limit: {loop_count}")
            continue_loop = False

        follow_up_message = handle_response(response_message, state)

        if follow_up_message is None:
            # No remaining work to do, return final response to user
            continue_loop = False
        else:
            messages.append(follow_up_message)


def chat_function(message: dict, history: list[dict], system_prompt: str, temperature: float, state: dict) -> Generator[str, None, None]:
    """
    Process a chat message and generate a response using an AI model.

    This function handles the main chat interaction, including processing the input message,
    formatting the conversation history, running the AI model loop, and generating a response.
    It also handles displaying additional content like images or text in the chat interface.

    Args:
        message (dict): The current message from the user.
        history (list): A list of previous messages in the conversation.
        system_prompt (str): The system prompt to guide the AI model's behavior.
        temperature (float): The temperature parameter for the AI model's response generation.
        state (dict): The current state of the chat interface.

    Yields:
        str: The generated response from the AI model, including any additional content.

    Note:
        This function modifies the 'state' dictionary to store output for display in the chat interface.
        It handles both text and file inputs, and can display generated images in the response.
    """
    if message['text'] == '':
        yield "Please enter a message."

    state['output'] = []
    messages = format_messages_for_bedrock_converse(message, history, state)
    manage_conversation_flow(messages, system_prompt, temperature, state)

    response = ''
    for output in state['output']:
        if output['format'] == 'text':
            additional_text = output['text']
            response += f"{additional_text}\n"
            history.append({"role": "assistant", "content": output['text']})
        else:
            image = output
            print(f"Showing image: {image['filename']}")
            response += f'<p><img alt="{escape(image["description"])}" src="file={image['filename']}"></p>\n'

    print() # Add an additional space
    yield response


def import_images(image_path: str) -> None:
    """
    Import images from the image_path directory and store them in the image catalog.

    This function scans the image_path directory for image files, processes each valid image,
    and stores it in the image catalog using the store_image function. It utilizes multithreading
    to improve performance when processing multiple images.

    The function creates the image_path directory if it doesn't exist, processes only files with
    extensions listed in IMAGE_FORMATS, and uses a ThreadPoolExecutor to parallelize the import process.

    Global variables:
    - opensearch_client: The OpenSearch client used for storing image metadata.
    - IMAGE_FORMATS: A list of valid image file extensions.
    - MAX_WORKERS: The maximum number of worker threads for parallel processing.

    Returns:
    None

    Side effects:
    - Creates the image_path directory if it doesn't exist.
    - Stores imported images in the image catalog.
    - Prints progress information to the console.
    """
    global opensearch_client

    print("Importing images...")

    def import_image(file):
        print(f"Found: {file}")
        file_name, extension = get_file_name_and_extension(file)
        if extension in IMAGE_FORMATS:
            image_base64 = get_image_base64(image_path + file)
            image = store_image(extension, image_base64, file_name)
            return image

    imported_images = []

    # Create IMAGES_PATH if not exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(import_image, file) for file in os.listdir(image_path)]

        for future in concurrent.futures.as_completed(futures):
            image = future.result()
            if image is not None:
                imported_images.append(image)

    print(f"Total images imported: {len(imported_images)}")


def invoke_lambda_function(function_name: str, event: dict) -> dict:
    """
    Invoke an AWS Lambda function and return its output.

    This function invokes a specified AWS Lambda function with the given event data,
    retrieves the response, and returns the decoded output.

    Args:
        function_name (str): The name or ARN of the Lambda function to invoke.
        event (dict): The event data to pass to the Lambda function.

    Returns:
        dict: The decoded output from the Lambda function.

    Note:
        This function uses the global lambda_client to make the API call.
    """
    global lambda_client

    response = lambda_client.invoke(
        FunctionName=function_name,
        Payload=json.dumps(event)
    )

    # Get output from response
    payload = response['Payload'].read().decode('utf-8')
    output = json.loads(payload)

    return output


def main(args: argparse.Namespace):
    """
    Main function to set up and run the chatbot application.

    This function performs the following tasks:
    1. Checks consistency of defined tools.
    2. Initializes the OpenSearch client.
    3. Handles index reset if requested.
    4. Creates necessary indexes.
    5. Imports images into the catalog.
    6. Sets up the Gradio chat interface with custom components.
    7. Launches the chat interface.

    Args:
        args (Namespace): Command-line arguments parsed by argparse.

    Note:
        This function uses several global variables and external functions
        for index management, image importing, and interface setup.
    """
    global opensearch_client

    check_tools_consistency()

    opensearch_client = get_opensearch_client()

    if args.reset_index:
        delete_index(opensearch_client, TEXT_INDEX_NAME)
        delete_index(opensearch_client, MULTIMODAL_INDEX_NAME)
        return

    create_index(opensearch_client, TEXT_INDEX_NAME, TEXT_INDEX_CONFIG)
    create_index(opensearch_client, MULTIMODAL_INDEX_NAME, MULTIMODAL_INDEX_CONFIG)

    import_images(IMAGE_PATH)

    print("Starting the chatbot...")

    state = gr.State({ "notebook": [], "notebook_current_page": 0, "output": [], "improvements": "" })

    # To enable the copy button
    custom_chatbot = gr.Chatbot(
        elem_id="chatbot",
        type="messages",
        label="Yet Another Chatbot",
        show_copy_button=True,
    )

    # To allow multiple file uploads
    custom_textbox = gr.MultimodalTextbox(
        placeholder="Enter your instructions and press enter.",
        file_count='multiple',
    )

    # Formatted for type "messages"
    formatted_examples = [
        [{"text": example, "files": []}] for example in EXAMPLES
    ]

    CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """

    chat_interface = gr.ChatInterface(
        css=CSS,
        fn=chat_function,
        type="messages",
        title="Yet Another Chatbot",
        description="Your Helpful AI Assistant. I can search and browse the web, search Wikipedia, the news, and maps, run Python code that I write, write long articles, and generate and compare images.",
        chatbot=custom_chatbot,
        textbox=custom_textbox,
        multimodal=True,
        examples=formatted_examples,
        examples_per_page=2,
        additional_inputs=[
            gr.Textbox(DEFAULT_SYSTEM_PROMPT, label="System Prompt"),
            gr.Slider(0, 1, value=DEFAULT_TEMPERATURE, label="Temperature"),
            state,
        ],
        fill_height=True,
    )

    abs_image_path = os.path.abspath(IMAGE_PATH)
    print(f"Allowed paths: {abs_image_path}")
    
    chat_interface.launch(allowed_paths=[abs_image_path])


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the chatbot application.

    This function sets up an argument parser to handle command-line options
    for the chatbot. It currently supports one optional flag:
    --reset-index: When set, this flag indicates that the text and multimodal
                   indexes should be reset. Note that image files are not deleted.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--reset-index', action='store_true', help='Reset text and multimodal indexes. Image files are not deleted.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
