#!/usr/bin/env python3

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import re
import urllib.request
import time

from datetime import datetime
from html import escape
from urllib.parse import urlparse
import urllib

import boto3
from botocore.config import Config

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

import numpy as np

# Fix for "Error: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead."
np.float_ = np.float64

# Fix to avoid the "The current process just got forked..." warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

IMAGE_PATH = "./Images/"
OUTPUT_PATH = "./Output/"

OPENSEARCH_HOST = 'localhost'
OPENSEARCH_PORT = 9200

MULTIMODAL_INDEX_NAME = "multimodal-index" 
TEXT_INDEX_NAME = "text-index" 
MAX_EMBEDDING_IMAGE_SIZE = 5 * 1024 * 1024 # 5MB
MAX_EMBEDDING_IMAGE_DIMENSIONS = 2048
MAX_INFERENCE_IMAGE_SIZE = 3.75 * 1024 * 1024 # 3.75MB
MAX_INFERENCE_IMAGE_DIMENSIONS = 8000
MAX_CHAT_IMAGE_SIZE = 1024 * 1024 # 1MB
MAX_CHAT_IMAGE_DIMENSIONS = 2048
JPEG_SAVE_QUALITY = 90

DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512

#AWS_REGION = 'us-east-1'
AWS_REGION = 'us-west-2'

AWS_LAMBDA_FUNCTION_NAME = "yet-another-chatbot-function"
MAX_OUTPUT_LENGTH = 4096 # 4K characters

EMBEDDING_MULTIMODAL_MODEL_ID = 'amazon.titan-embed-image-v1'
EMBEDDING_TEXT_MODEL_ID = 'amazon.titan-embed-text-v2:0'

IMAGE_GENERATION_MODEL_UD = 'amazon.titan-image-generator-v2:0'
#IMAGE_GENERATION_MODEL_UD = 'amazon.titan-image-generator-v1'

# True to handle document-to-text translation in code instead of using the Bedrock Converse API Document Chat capability
HANDLE_DOCUMENT_TO_TEXT_IN_CODE = True

MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
#MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
#MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'
#MODEL_ID = 'meta.llama3-1-405b-instruct-v1:0'
#MODEL_ID = 'meta.llama3-1-70b-instruct-v1:0'
#MODEL_ID = 'meta.llama3-1-8b-instruct-v1:0'
#MODEL_ID = 'mistral.mistral-large-2407-v1:0' # Mistral Large 2 (24.07)

MIN_RETRY_WAIT_TIME = 5 # seconds
MAX_RETRY_WAIT_TIME = 40 # seconds

MAX_TOKENS = 4096
#MAX_TOKENS = 2048 # For some models

MAX_LOOPS = 128

MAX_WORKERS = 10

MIN_CHUNK_LENGTH = 800    
MAX_CHUNK_LENGTH = 900

MAX_SEARCH_RESULTS = 10
MAX_ARCHIVE_RESULTS = 10
MAX_IMAGE_SEARCH_RESULTS = 3

DEFAULT_TEMPERATURE = 0.5

DEFAULT_SYSTEM_PROMPT = """You are an helpful AI assistant. Think step-by-step. Build a plan. Follow instructions carefully.
Use the appropriate tool for a task. Use more than one tool to provide the best answer.
As first step, always compute and replace relative dates (today, tomorrow, last week, this year) into exact dates (January 1, 1970).
Before searching or browsing the internet, ask questions to check if you have information in the archive to improve your answer.
Always browse the actual websites and URLs to get updated and detailed info.
Use existing images when possible and don't generate new images unless really necessary.
Don't mention the name of the tools to the user.
Never show or tell an 'image_id' to the user. Don't use the term 'image_id' with users."""

IMAGE_FORMATS = ['png', 'jpeg', 'gif', 'webp']
DOCUMENT_FORMATS = ['pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md']

IMAGE_DESCRIPTION_PROMPT = "Describe this image in 50 words or less."
IMAGE_FILTER_PROMPT = "Remove from this JSON list the images that don't match the description. Only output JSON and nothing else."

TOOLS_TIMEOUT = 300  # Timeout in seconds

opensearch_client = None

# AWS SDK for Python (Boto3) clients
bedrock_runtime_config = Config(connect_timeout=300, read_timeout=300, retries={'max_attempts': 4})
bedrock_runtime_client = boto3.client('bedrock-runtime', region_name=AWS_REGION, config=bedrock_runtime_config)
iam_client = boto3.client('iam', region_name=AWS_REGION)
lambda_client = boto3.client('lambda', region_name=AWS_REGION)


def load_json_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


TOOLS = load_json_config("./Config/tools.json")
TEXT_INDEX_CONFIG = load_json_config("./Config/text_vector_index.json")
MULTIMODAL_INDEX_CONFIG = load_json_config("./Config/multimodal_vector_index.json")
EXAMPLES = load_json_config('./Config/examples.json')


def add_as_output(content, state):
    # Avoid to show duplicate images
    if 'image_id' in content:
        image = content
        for item in state['output']:
            if 'image_id' in item and item['image_id'] == image['image_id']:
                return
    state['output'].append(content)


def get_opensearch_client():
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


def delete_index(client, index_name):
    if client.indices.exists(index=index_name):
        try:
            _ = client.indices.delete(
                index=index_name,
            )
            print(f"Index {index_name} deleted.")
        except Exception as ex: print(ex)


def create_index(client, index_name, index_config):
    if not client.indices.exists(index=index_name):
        try:
            _ = client.indices.create(
                index=index_name,
                body=index_config,
            ) 
            print(f"Index {index_name} created.")
        except Exception as ex: print(ex)


def print_index_info():
    global opensearch_client
    try:
        response = opensearch_client.indices.get(MULTIMODAL_INDEX_NAME) 
        print(json.dumps(response, indent=2))
    except Exception as ex: print(ex)


def get_image_bytes(image_source, max_image_size=None, max_image_dimension=None):
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


def get_image_base64(image_source, max_image_size=None, max_image_dimension=None):
    image_bytes = get_image_bytes(image_source, max_image_size, max_image_dimension)
    return base64.b64encode(image_bytes).decode('utf-8')


def get_embedding(image_base64:str=None, input_text:str=None, multimodal:str=False):
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


def invoke_text_model(messages, system_prompt=None, temperature=0, tools=None, return_last_message_only=False):
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
        # Add current date and time to the system prompt
        current_date_and_day_of_the_week = datetime.now().strftime("%a, %Y-%m-%d")
        current_time = datetime.now().strftime("%I:%M:%S %p")
        system_prompt += f"\nWhen processing dates and times, consider that today is {current_date_and_day_of_the_week} and the current time is {current_time}."
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


def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    return base_url


def mark_down_formatting(html_text: str, url: str):
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


def remove_xml_tags(text: str):
    cleaned_text = text

    # Remove self reflection XML tags and their content
    for tag in ['search_quality_reflection', 'search_quality_score', 'image_quality_score']:
        cleaned_text = re.sub(fr'<{tag}>.*?</{tag}>', '', cleaned_text, flags=re.DOTALL)

    # Remove all XML tags (but not their content) excluding comments (starting with "<!")
    cleaned_text = re.sub(r'<[^!>]+>', '', cleaned_text)

    return cleaned_text


def with_xml_tag(text: str, tag: str) -> str:
    return f"<{tag}>{text}</{tag}>"


def split_text_for_collection(text: str):
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


def add_to_text_index(text: str, id: str, metadata: dict, metadata_delete: dict|None=None):
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


def add_to_multimodal_index(image: dict, image_base64: str):
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


def get_image_hash(image_bytes):
    hash_obj = hashlib.sha256()
    hash_obj.update(image_bytes)
    return hash_obj.hexdigest()


def store_image(image_format: str, image_base64: str, import_image_id:str=''):
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


def get_image_by_id(image_id: str, return_base64=False) -> dict|str:
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


def search_images(query: str, index_name: str) -> list|str:
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


def get_images_by_description(description: str, max_results: int):
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

def get_images_by_similarity(image_id: str, max_results: int):
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


def get_random_images(num: int):
    query = {
        "size": num,
        "query": { "function_score": { "random_score": {} } },
        "_source": ["format", "filename", "description"]
    }
    images = search_images(query, MULTIMODAL_INDEX_NAME)
    return images


def get_tool_result_python(tool_input, state):
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
    add_as_output(
        { "format": "text", "text": f"Python script:\n\n```python\n{input_script}\n```\n" },
        state
    )
    add_as_output(
         { "format": "text", "text": f"Output:\n\n```\n{output}\n```\n" },
         state
    )
    print(f"Output:\n---\n{output}\n---\nThe script and its output will be shared with the user at the of your message.")

    return f"{with_xml_tag(output, 'output')}"


def get_tool_result_duckduckgo_text(tool_input, _state):
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


def get_tool_result_duckduckgo_news(tool_input, _state):
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


def get_tool_result_duckduckgo_maps(tool_input, _state):
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


def get_tool_result_wikipedia_search(tool_input, _state):
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


def get_tool_result_wikipedia_geodata_search(tool_input, _state):
    latitude = tool_input["latitude"]
    longitude = tool_input["longitude"]
    search_title = tool_input.get("title", None)  # Optional
    radius = tool_input.get("radius", None)  # Optional
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


def get_tool_result_wikipedia_page(tool_input, _state):
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


def get_tool_result_browser(tool_input, _state):
    url = tool_input["url"]
    print(f"URL: {url}")

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as f:
            status = f.status
            content_length = f.headers.get("Content-Length", 0)
        print(f"Status: {status}")
        print(f"Content length: {content_length}")
        if status >= 400:
            return f"HTTP status {status}"
    except Exception as e:
        print(f"Error: {e}")

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


def get_tool_result_retrive_from_archive(tool_input, _state):
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


def get_tool_result_store_in_archive(tool_input, _state):
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
    rendered_notebook = "\n\n".join(notebook)
    rendered_notebook = re.sub(r'\n{3,}', '\n\n', rendered_notebook)
    return rendered_notebook

def get_tool_result_notebook(tool_input, state):
    command = tool_input["command"]
    print(f"Command: {command}")
    content = tool_input.get("content", "")
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
            add_as_output({"format": "text", "text": notebook_output}, state)
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


def get_tool_result_generate_image(tool_input, state):
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


def get_tool_result_search_image_catalog(tool_input, state):
    description = tool_input.get("description", None)
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


def get_tool_result_similarity_image_catalog(tool_input, state):
    image_id = tool_input.get("image_id", None)
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


def get_tool_result_random_images(tool_input, state):
    num = tool_input.get("num", None)
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


def get_tool_result_image_catalog_count(_tool_input, _state):
    try:
        info = opensearch_client.count(index=MULTIMODAL_INDEX_NAME)
        print(f"Image catalog info: {info}")
        count = info["count"]
        return count
    except Exception as ex:
        error_message = f"Error: {ex}"
        print(error_message)
        return error_message


def get_tool_result_download_image_into_catalog(tool_input, state):
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


class ToolError(Exception):
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
}


def check_tools_consistency():
    tools_set = set([ t['toolSpec']['name'] for t in TOOLS])
    tool_functions_set = set(TOOL_FUNCTIONS.keys())

    if tools_set != tool_functions_set:
        raise Exception(f"Tools and tool functions are not consistent: {tools_set} != {tool_functions_set}")


def get_tool_result(tool_use_block, state):
    global opensearch_client

    tool_use_name = tool_use_block['name']
            
    print(f"Using tool {tool_use_name}")
    
    try:
        return TOOL_FUNCTIONS[tool_use_name](tool_use_block['input'], state)
    except KeyError:
        raise ToolError(f"Invalid function name: {tool_use_name}")


def handle_response(response_message, state):
    
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


def get_file_name_and_extension(full_file_name: str):
    file_name, extension = os.path.splitext(os.path.basename(full_file_name))
    if len(extension) > 0:
        extension = extension[1:].lower() # Remove the leading '.' and make it lowercase
    return file_name, extension


def format_messages_for_bedrock_converse(message, history, state):

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
            m_text = remove_xml_tags(m_text) # To remove <img> tags
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


def run_loop(messages, system_prompt, temperature, state):
    loop_count = 0
    continue_loop = True
    
    new_messages = []

    while continue_loop:

        response = invoke_text_model(messages, system_prompt, temperature, tools=TOOLS)
        
        response_message = response['output']['message']
        messages.append(response_message)
        new_messages.append(response_message)
        
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
            new_messages.append(follow_up_message)
    
    assistant_responses = []
    for m in new_messages:
        if m['role'] == 'assistant' and 'content' in m:
            for c in m['content']:
                if 'text' in c:
                    assistant_responses.append(c['text'])
                if 'toolUse' in c:
                    hidden_content = "<!--\n" + json.dumps(c['toolUse']) + "\n-->"
        if m['role'] == 'user' and 'content' in m:
            for c in m['content']:
                if 'toolResult' in c:
                    tool_result = ''
                    content = c['toolResult']['content'][0]
                    if 'json' in content:
                        tool_result = json.dumps(content['json'])
                    elif 'text' in content:
                        tool_result = content['text']
                    if len(tool_result) > 0:
                        # Hidden to not show up in the chat but still be used by the model
                        hidden_content = "<!-- " + tool_result + " -->"
                        assistant_responses.append(hidden_content)

    return remove_xml_tags("\n".join(assistant_responses))


def chat_function(message, history, system_prompt, temperature, state):

    if message['text'] != '':
        state['output'] = []
        messages = format_messages_for_bedrock_converse(message, history, state)
        response = run_loop(messages, system_prompt, temperature, state)
        print(f"Response length: {len(response)}")

        if len(state['output']) > 0:
            for output in state['output']:
                if output['format'] == 'text':
                    additional_text = output['text']
                    print(f"Additional text length: {len(additional_text)}")
                    response += f"\n{additional_text}"
                    history.append({"role": "assistant", "content": output['text']})
                else:
                    image = output
                    print(f"Showing image: {image['filename']}")
                    response += f'<p><img alt="{escape(image["description"])}" src="file={image['filename']}"></p>'
            print(f"Response length with additional content: {len(response)}")

        print() # Add an additional space
        yield response

    else:
        yield "Please enter a message."


def import_images():
    global opensearch_client

    print("Importing images...")

    def import_image(file):
        print(f"Found: {file}")
        file_name, extension = get_file_name_and_extension(file)
        if extension in IMAGE_FORMATS:
            image_base64 = get_image_base64(IMAGE_PATH + file)
            image = store_image(extension, image_base64, file_name)
            return image

    imported_images = []

    # Create IMAGES_PATH if not exists
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True) 

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(import_image, file) for file in os.listdir(IMAGE_PATH)]

        for future in concurrent.futures.as_completed(futures):
            image = future.result()
            if image is not None:
                imported_images.append(image)

    print(f"Total images imported: {len(imported_images)}")


def invoke_lambda_function(function_name, event):
    global lambda_client

    response = lambda_client.invoke(
        FunctionName=function_name,
        Payload=json.dumps(event)
    )

    # Get output from response
    payload = response['Payload'].read().decode('utf-8')
    output = json.loads(payload)

    return output


def main(args):
    global opensearch_client

    check_tools_consistency()

    opensearch_client = get_opensearch_client()

    if args.reset_index:
        delete_index(opensearch_client, TEXT_INDEX_NAME)
        delete_index(opensearch_client, MULTIMODAL_INDEX_NAME)
        return

    create_index(opensearch_client, TEXT_INDEX_NAME, TEXT_INDEX_CONFIG)
    create_index(opensearch_client, MULTIMODAL_INDEX_NAME, MULTIMODAL_INDEX_CONFIG)

    import_images()

    print("Starting the chatbot...")

    state = gr.State({ "notebook": [], "notebook_current_page": 0, "output": [] })

    # To enable the copy button
    custom_chatbot = gr.Chatbot(
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

    chat_interface = gr.ChatInterface(
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
    )


    abs_image_path = os.path.abspath(IMAGE_PATH)
    chat_interface.launch(allowed_paths=[abs_image_path])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--reset-index', action='store_true', help='Reset text and multimodal indexes. Image files are not deleted.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
