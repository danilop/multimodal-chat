import base64
import concurrent.futures
from contextlib import closing
import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.request

from html import escape

import chardet
from PIL import Image
from opensearchpy import NotFoundError
from opensearchpy.helpers import bulk
import pypandoc
from pypdf import PdfReader

from libs import between_xml_tag
from config import Config
from clients import Clients
from usage import ModelUsage


class ImageNotFoundError(Exception):
        pass


class Utils:

    def __init__(self, config: Config, clients: Clients):
        self.config = config
        self.clients = clients
        self.usage = ModelUsage()

    def get_embedding(self, image_base64: str | None = None, input_text: str | None = None, multimodal: bool = False) -> list[float] | None:
        """
        Generate an embedding vector for the given image and/or text input using Amazon Bedrock.

        This method can handle text-only, image-only, or multimodal (text + image) inputs.
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
        use_multimodal = False
        body = {}
        if input_text is not None:
            body["inputText"] = input_text

        if image_base64 is not None:
            body["inputImage"] = image_base64

        if multimodal or 'inputImage' in body:
            embedding_model_id = self.config.EMBEDDING_MULTIMODAL_MODEL_ID
            use_multimodal = True
        elif 'inputText' in body:
            embedding_model_id = self.config.EMBEDDING_TEXT_MODEL_ID
        else:
            return None

        response = self.clients.bedrock_runtime_client.invoke_model(
            body=json.dumps(body),
            modelId=embedding_model_id,
            accept="application/json", contentType="application/json",
        )

        response_body = json.loads(response.get('body').read())
        finish_reason = response_body.get('message')
        if finish_reason is not None:
            print(finish_reason)
            print(f"Body: {body}")
        embedding_vector = response_body.get('embedding')

        input_text_token_count = response_body.get('inputTextTokenCount')
        if use_multimodal:
            self.usage.update('inputMultimodalTokenCount', input_text_token_count)
        else:
            self.usage.update('inputTextTokenCount', input_text_token_count)
        if 'inputImage' in body:
            self.usage.update('inputImageCount', 1)

        return embedding_vector

    def add_to_multimodal_index(self, image: dict, image_base64: str) -> None:
        """
        Add an image and its metadata to the multimodal index in OpenSearch.

        This method computes an embedding vector for the image and its description,
        then indexes this information along with other image metadata in OpenSearch.
        It uses multimodal capabilities to create a combined embedding of the image and its textual description.

        Args:
            image (dict): A dictionary containing image metadata including:
                - format (str): The image format (e.g., 'png', 'jpeg')
                - filename (str): The name of the image file
                - description (str): A textual description of the image
                - id (str): A unique identifier for the image
            image_base64 (str): The base64-encoded string representation of the image

        Raises:
            Exception: If there's an error during the indexing process

        Note:
            This method assumes that the OpenSearch client is properly initialized and configured.
            The index name is determined by the configuration (self.config.MULTIMODAL_INDEX_NAME).
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
        embedding_vector = self.get_embedding(image_base64=image_base64, input_text=image['description'])
        document = {
            "format": image['format'],
            "filename": image['filename'],
            "description": image['description'],
            "embedding_vector": embedding_vector,
        }
        response = self.clients.opensearch_client.index(
            index=self.config.MULTIMODAL_INDEX_NAME,
            body=document,
            id=image['id'],
            refresh=True,
        )
        print(f"Multimodel index result: {response['result']}")

    def store_image(self, image_format: str, image_base64: str, import_image_id:str=''):
        """
        Store an image in the file system and index it in the multimodal database.

        This method takes a base64-encoded image, stores it in the file system,
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
        image_id = self.get_image_hash(image_bytes)

        if import_image_id != '' and import_image_id != image_id:
            print(f"Image ID mismatch: {import_image_id} != computed {image_id }")
            return None

        image_filename = self.config.IMAGE_PATH + image_id + '.' + image_format
        if not os.path.exists(image_filename):
            with open(image_filename, 'wb') as f:
                f.write(image_bytes)

        image = self.get_image_by_id(image_id)
        if type(image) is dict:
            print("Image already indexed.")
            return image
        
        # Short description to fit in multimodal embeddings
        image_description = self.get_image_description(image_bytes, image_format)

        image = {
            "id": image_id,
            "format": image_format,
            "filename": image_filename,
            "description": image_description,
        }

        self.add_to_multimodal_index(image, image_base64)

        return image

    def invoke_lambda_function(self, function_name: str, event: dict) -> dict:
        """
        Invoke an AWS Lambda function and return its output.

        This method invokes a specified AWS Lambda function with the given event data,
        retrieves the response, and returns the decoded output.

        Args:
            function_name (str): The name or ARN of the Lambda function to invoke.
            event (dict): The event data to pass to the Lambda function.

        Returns:
            dict: The decoded output from the Lambda function.

        Note:
            This function uses the global lambda_client to make the API call.
        """
        try:
            response = self.clients.lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(event)
            )
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return { "output": error_message }

        # Get output from response
        payload = response['Payload'].read().decode('utf-8')
        body = json.loads(payload).get('body', '{}')
        result = json.loads(body) # Contains the output (str) and images (list) keys

        return result

    def get_image_bytes(self, image_source: str | bytes, format: str = "JPEG", max_image_size: int | None = None, max_image_dimension: int | None = None) -> bytes:
        """
        Retrieve image bytes from a source and optionally resize the image.

        This method can handle both URL and local file path sources. It will
        resize the image if it exceeds the specified maximum size or dimension.

        Args:
            image_source (str): URL, local path, or base64-encoded string of the image.
            format (str, optional): Image format to use when saving the image. Defaults to "JPEG".
            max_image_size (int, optional): Maximum allowed size of the image in bytes.
            max_image_dimension (int, optional): Maximum allowed dimension (width or height) of the image.

        Returns:
            bytes: The image data as bytes, potentially resized.

        Note:
            If resizing is necessary, the function will progressively reduce the image size
            until it meets the specified constraints. The resized image is saved in JPEG format.
        """

        def check_if_base64_image(image_source: any) -> io.BytesIO | None:
            if not isinstance(image_source, str):
                return None

            try:
                decoded_data = base64.b64decode(image_source)
                original_image_bytes = io.BytesIO(decoded_data)
                Image.open(original_image_bytes) # This fails if the image is not valid
                return original_image_bytes
            except:
                return None

        image_bytes = check_if_base64_image(image_source)
        
        if image_bytes is None:
            if isinstance(image_source, bytes):
                image_bytes = io.BytesIO(image_source)  
            elif image_source.startswith(('http://', 'https://')):
                print(f"Downloading image from URL: {image_source}")
                # Download image from URL
                with urllib.request.urlopen(image_source) as response:
                    image_bytes = io.BytesIO(response.read())
            else:
                # Open image from local path
                print(f"Opening image from local path: {image_source}")
                image_bytes = io.BytesIO()
                with open(image_source, 'rb') as f:
                    image_bytes.write(f.read())
        else:
            print("Image is base64 encoded.")

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
                    img.save(image_bytes, format=format)
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

    def get_image_base64(self, image_source: str, format: str = "JPEG", max_image_size: int | None = None, max_image_dimension: int | None = None) -> str:
        """
        Convert an image to a base64-encoded string, with optional resizing.

        Args:
            image_source (str): URL or local path of the image.
            format (str, optional): Image format to use when saving the image. Defaults to "JPEG".
            max_image_size (int, optional): Maximum allowed size of the image in bytes.
            max_image_dimension (int, optional): Maximum allowed dimension (width or height) of the image.

        Returns:
            str: Base64-encoded string representation of the image.

        Note:
            This function uses get_image_bytes to retrieve and potentially resize the image
            before encoding it to base64.
        """
        image_bytes = self.get_image_bytes(image_source, format, max_image_size, max_image_dimension)
        return base64.b64encode(image_bytes).decode('utf-8')

    def get_image_hash(self, image_bytes: bytes) -> str:
        """
        Compute a SHA-256 hash for the given image bytes.

        This method takes the raw bytes of an image and computes a unique
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


    def get_image_description(self, image_bytes: bytes, image_format: str, detailed: bool = False) -> str:
        """
        Generate a description for an image using the AI model.

        This method uses an AI model to analyze the given image and generate a textual description.
        The description can be either brief or detailed based on the 'detailed' parameter.

        Args:
            image_bytes (bytes): The raw bytes of the image to be described.
            image_format (str): The format of the image (e.g., 'png', 'jpeg', 'gif').
            detailed (bool, optional): If True, generate a more comprehensive and detailed description.
                                       If False, generate a brief description. Defaults to False.

        Returns:
            str: A textual description of the image generated by the AI model.

        Note:
            The quality and accuracy of the description depend on the capabilities of the underlying AI model.
        """
        if detailed:
            prompt = self.config.DETAILED_IMAGE_DESCRIPTION_PROMPT
        else:
            prompt = self.config.SHORT_IMAGE_DESCRIPTION_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_bytes}
                    }
                },
                {"text": prompt}
            ],
        }]

        print(f"Generating {'detailed ' if detailed else ''}image description...")
        image_description = self.invoke_text_model(messages, return_last_message_only=True)
        print(f"Image description: {image_description}")

        return image_description

    def get_image_by_id(self, image_id: str, return_base64: bool = False) -> dict|str:
        """
        Retrieve image metadata from the multimodal index by its ID.

        This method queries the OpenSearch index to fetch metadata for an image
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
            response = self.clients.opensearch_client.get(
                id=image_id,
                index=self.config.MULTIMODAL_INDEX_NAME,
                _source_includes=["format", "filename", "description"],
            )
            image = response['_source']
            image['id'] = image_id
            if return_base64:
                image['base64'] = self.get_image_base64(image['filename'], format=image['format'])
            return image
        except NotFoundError:
            return "Not found."
        except Exception as ex:
            error_message = f"Error: {ex}"
            print(error_message)
            return error_message

    def invoke_text_model(self, messages: list[dict], system_prompt: str | None = None, temperature: float = 0, tools: list[dict] | None = None, return_last_message_only: bool = False) -> dict | str:
        """
        Invoke the text model using Amazon Bedrock's converse API.

        This method prepares the request body, handles retries for throttling exceptions,
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
        converse_body = {
            "modelId": self.config.MODEL_ID,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.config.MAX_TOKENS,
                "temperature": temperature,
                },
        }

        if system_prompt is not None:
            converse_body["system"] = [{"text": system_prompt}]

        if tools:
            converse_body["toolConfig"] = {"tools": tools}

        print("Thinking...")

        # To handle throttling retries
        retry_wait_time = self.config.MIN_RETRY_WAIT_TIME
        retry_flag = True

        while(retry_flag and retry_wait_time <= self.config.MAX_RETRY_WAIT_TIME):
            try:
                response = self.clients.bedrock_runtime_client.converse(**converse_body)
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

        print(f"Stop reason: {response['stopReason']}")

        for metrics, value in response['usage'].items():
            self.usage.update(metrics, value)
        print(self.usage)

        if return_last_message_only:
            response_message = response['output']['message']
            last_message = response_message['content'][0]['text']
            return last_message

        return response

    def add_to_text_index(self, text: str, id: str, metadata: dict, metadata_delete: dict|None=None) -> None:
        """
        Add text content to the text index in OpenSearch.

        This method processes the input text, splits it into chunks, computes embeddings,
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
        if metadata_delete is not None:
            # Delete previous content
            delete_query = {
                "query": {
                    "match": metadata_delete
                }
            }
            response = self.clients.opensearch_client.delete_by_query(
                index=self.config.TEXT_INDEX_NAME,
                body=delete_query,
            )
            deleted = response['deleted']
            if deleted > 0:
                print(f"Deleted old content: {deleted}")

        def process_chunk(i, chunk, metadata, id):
            formatted_metadata = '\n '.join([f"{key}: {value}" for key, value in metadata.items()])
            chunk = f"{formatted_metadata}\n\n{chunk}"
            text_embedding = self.get_embedding(input_text=chunk)
            document = {
                "id": f"{id}_{i}",
                "document": chunk,
                "embedding_vector": text_embedding,
            }
            document = document | metadata
            return document

        chunks = self.split_text_for_collection(text)
        print(f"Split into {len(chunks)} chunks")

        # Compute embeddings
        avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        min_chunk_length = min(len(chunk) for chunk in chunks)
        max_chunk_length = max(len(chunk) for chunk in chunks)
        print(f"Embedding {len(chunks)} chunks with min/average/max {min_chunk_length}/{round(avg_chunk_length)}/{max_chunk_length} characters...")
        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [executor.submit(process_chunk, i + 1, chunk, metadata, id) for i, chunk in enumerate(chunks)]

            for future in concurrent.futures.as_completed(futures):
                document = future.result()
                documents.append(document)
        print(f"Indexing {len(documents)} chunks...")

        success, failed = bulk(
            self.clients.opensearch_client,
            documents,
            index=self.config.TEXT_INDEX_NAME,
            raise_on_exception=True
        )
        print(f"Indexed {success} documents successfully, {len(failed)} documents failed.")

    def split_text_for_collection(self, text: str) -> list[str]:
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
            if len(chunk) < self.config.MAX_CHUNK_LENGTH:
                chunk += sentence + "\n"
                if len(chunk) > self.config.MIN_CHUNK_LENGTH:
                    next_chunk += sentence + "\n"
            else:
                if len(chunk) > 0:
                    chunks.append(chunk)
                chunk = next_chunk
                next_chunk = ''

        if len(chunk) > 0:
            chunks.append(chunk)

        return chunks

    def process_image_placeholders(self, text: str, for_output_file: bool = False) -> str:
        """
        Replace image placeholders with markdown to display the image.

        Args:
            page (str): A string representing a page in the sketchbook.

        Returns:
            str: The page with image placeholders replaced by markdown.
        """
        def replace_image(match):
            image_id = match.group(1)
            image = self.get_image_by_id(image_id)
            if isinstance(image, dict):
                filename = image["filename"]
                if for_output_file:
                    filename = os.path.relpath(os.path.join('..', filename))
                    # Using normal Markdown syntax
                    return f'![{escape(image["description"])}]({filename})'
                # Using Gradio syntax with "file="
                return f'![{escape(image["description"])}](file={filename})'
            else:
                error_message = f"Image with 'image_id' {image_id} not found in the image catalog."
                print(error_message)
                raise ImageNotFoundError(error_message)

        return re.sub(r'\[image_id:\s*(\w+)\]', replace_image, text)

    def get_file_name_and_extension(self, full_file_name: str) -> tuple[str, str]:
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

    def is_text_file(self, file_path: str) -> bool:
        """
        Check if a file is likely to be a text file based on its content.

        This function uses chardet to detect the encoding of the file.
        If an encoding is detected with high confidence, it's likely a text file.

        Args:
            file_path (str): The path to the file to be checked.

        Returns:
            bool: True if the file is likely to be a text file, False otherwise.
        """
        # Read a sample of the file content
        sample_size = 1024  # Adjust this value as needed
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
        except IOError:
            return False

        # If the sample is empty, consider it as non-text
        if not raw_data:
            return False

        # Use chardet to detect the encoding
        result = chardet.detect(raw_data)

        # Check if an encoding was detected with high confidence
        if result['encoding'] is not None and result['confidence'] > 0.7:
            return True

        return False

    def process_pdf_document(self, file: str) -> str:
        """
        Process a PDF document and extract text and images.

        This function uses the PyPDF library to extract text from a PDF file.
        It also processes images within the PDF and stores them in the image catalog when HANDLE_IMAGES_IN_DOCUMENTS is True.

        Args:
            file (str): The path to the PDF file to be processed.
            output_queue (queue.Queue): A queue to put the output into.

        Returns:
            str: The text content of the PDF file.
        """
        text_pages = []
        reader = PdfReader(file)
        for index, page in enumerate(reader.pages):
            text = page.extract_text((0, 90))
            if self.config.HANDLE_IMAGES_IN_DOCUMENTS:
                for image in page.images:
                    print(f"Processing image: {image.name}")
                    image_format = image.name.split('.')[-1].lower()
                    if image_format == 'jpg': # Quick fix
                        image_format = 'jpeg'
                    if image_format in self.config.IMAGE_FORMATS:
                        detailed_description = self.get_image_description(image.data, image_format, detailed=True)
                        image_base64 = self.get_image_base64(
                            image.data,
                            format=image_format,
                            max_image_size=self.config.MAX_CHAT_IMAGE_SIZE,
                            max_image_dimension=self.config.MAX_CHAT_IMAGE_DIMENSIONS
                        )
                        stored_image = self.store_image(image_format, image_base64)
                        text += "\n" + between_xml_tag(f"Name '{image.name}':\nImage store image_id: {stored_image['id']}\nDetailed description: {detailed_description}", 'image')
            text_pages.append(between_xml_tag(text, 'page', {'id': index}))
        return "\n".join(text_pages)

    def process_non_pdf_documents(self, file: str) -> str:
        """
        Process a non-PDF document and extract text and images.

        This function uses the pypandoc library to extract text from a PDF file.
        It also processes images within the PDF and stores them in the image catalog when HANDLE_IMAGES_IN_DOCUMENTS is True.

        Args:
            file (str): The path to the PDF file to be processed.
            output_queue (queue.Queue): A queue to put the output into.

        Returns:
            str: The text content of the PDF file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_text = pypandoc.convert_file(
                file,
                'rst',
                extra_args=[f'--extract-media={temp_dir}']
            )
            if self.config.HANDLE_IMAGES_IN_DOCUMENTS:
                for root, _dirs, files in os.walk(temp_dir):
                    for file in files:
                        format = file.split('.')[-1].lower()
                        if format == 'jpg': # Quick fix
                            format = 'jpeg'
                        if format in self.config.IMAGE_FORMATS:
                            image_path = os.path.join(root, file)
                            with open(image_path, 'rb') as img_file:
                                image_bytes = img_file.read()
                            detailed_description = self.get_image_description(image_bytes, format, detailed=True)
                            image_base64 = self.get_image_base64(
                                image_bytes,
                                format=format,
                                max_image_size=self.config.MAX_CHAT_IMAGE_SIZE,
                                max_image_dimension=self.config.MAX_CHAT_IMAGE_DIMENSIONS
                            )
                            image = self.store_image(format, image_base64)
                            file_text += f"\n\nExtracted image:\nimage_id: {image['id']}\ndescription: {detailed_description}"
        return file_text

    def synthesize_speech(self, text: str, voice: str) -> str:
        """
        Synthesize speech from text using Amazon Polly.

        This function uses the boto3 library to synthesize speech from the given text
        using the specified voice. The synthesized speech is returned as a base64-encoded
        string.

        Args:
            text (str): The text to be synthesized.
            voice (str): The voice to be used for synthesis.

        Returns:
            str: The base64-encoded synthesized speech.
        """
        response = self.clients.polly_client.synthesize_speech(
            Engine='generative',
            OutputFormat='mp3',
            VoiceId=voice,
            Text=text
        )
        with closing(response["AudioStream"]) as audio_stream:
            audio_data = audio_stream.read()
        
        return audio_data
    
    def delete_index(self, index_name: str) -> None:
        """
        Delete an index from OpenSearch if it exists.

        Args:
            client (OpenSearch): The OpenSearch client.
            index_name (str): The name of the index to be deleted.

        Note:
            This function prints the result of the deletion attempt or any exception that occurs.
        """
        if self.clients.opensearch_client.indices.exists(index=index_name):
            try:
                _ = self.clients.opensearch_client.indices.delete(
                    index=index_name,
                )
                print(f"Index {index_name} deleted.")
            except Exception as ex: print(ex)


    def create_index(self, index_name: str, index_config: dict) -> None:
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
        if not self.clients.opensearch_client.indices.exists(index=index_name):
            try:
                _ = self.clients.opensearch_client.indices.create(
                    index=index_name,
                    body=index_config,
                )
                print(f"Index {index_name} created.")
            except Exception as ex: print(ex)


    def print_index_info(self, index_name: str) -> None:
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
            response = self.clients.opensearch_client.indices.get(index=index_name)
            print(json.dumps(response, indent=2))
        except Exception as ex: print(ex)


    def replace_specific_xml_tags(self, text: str) -> str:
        """
        Replace specific XML tags with their content.

        This function replaces specified XML tags with their content.
        It targets specific tags related to quality scores and reflections.
        it replaces both opening and closing tags.

        Args:
            text (str): The input text containing XML tags.

        Returns:
            str: The text with specified XML tags replaced by their content.

        Note:
            This function uses regular expressions for tag replacement, which may not be suitable for
            processing very large XML documents due to performance considerations.
        """
        cleaned_text = text

        tag_to_replace = {
            'thinking': 'small',
        }

        # Replace specific XML tags with their content
        for tag, replacement in tag_to_replace.items():
            def replace_tag(tag, pattern, cleaned_text):
                return re.sub(pattern, lambda m: f"<{replacement}>{m.group(1)}</{replacement}>", cleaned_text, flags=re.DOTALL)

            # Replace both opening and closing tags
            pattern = fr'<{tag}>(.*?)</{tag}>'
            cleaned_text = replace_tag(tag, pattern, cleaned_text)
            
            # Replace unclosed tags (opening tag without closing tag)
            unclosed_pattern = fr'<{tag}>(.*?)(?=<|$)'
            cleaned_text = replace_tag(tag, unclosed_pattern, cleaned_text)
        return cleaned_text

    def remove_specific_xml_tags(self, text: str) -> str:
        """
        Remove specific XML tags and their content from the input text and return a dictionary of removed content.

        This function removes specified XML tags and their content from the input text.
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

        tags_to_remove = ['search_quality_reflection', 'search_quality_score', 'image_quality_score'] # , 'thinking']:

        # Remove specific XML tags and their content
        for tag in tags_to_remove:
            def remove_tag(tag, pattern, cleaned_text):
                return re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

            # Remove closed tags
            pattern = fr'<{tag}>(.*?)</{tag}>'
            cleaned_text = remove_tag(tag, pattern, cleaned_text)
            
            # Remove unclosed tags
            unclosed_pattern = fr'<{tag}>(.*?)$'
            cleaned_text = remove_tag(tag, unclosed_pattern, cleaned_text)

        return cleaned_text


    def search_images(self, query: str, index_name: str) -> list[dict]|str:
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
            response = self.clients.opensearch_client.search(
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


    def get_images_by_description(self, description: str, max_results: int) -> list[dict]:
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
        multimodal_text_embedding = self.get_embedding(input_text=description, multimodal=True)
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
        images = self.search_images(query, self.config.MULTIMODAL_INDEX_NAME)

        print("Filtering results...")

        prompt = (self.config.IMAGE_FILTER_PROMPT + '\n' +
                between_xml_tag(json.dumps(images), 'images') + '\n' +
                between_xml_tag(description, 'description'))

        messages = [{
            "role": "user",
            "content": [ { "text": prompt } ],
        }]

        response_message = self.invoke_text_model(messages, return_last_message_only=True)
        filtered_images = json.loads(response_message)

        print(f"From {len(images)} to {len(filtered_images)} images.")

        return filtered_images

    def get_images_by_similarity(self, image_id: str, max_results: int) -> list[dict]:
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
        image = self.get_image_by_id(image_id, return_base64=True)
        if type(image) is not dict:
            return "Image not found."
        multimodal_embedding = self.get_embedding(
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
        similar_images = self.search_images(query, self.config.MULTIMODAL_INDEX_NAME)
        if type(similar_images) is list and len(similar_images) > 0:
            similar_images.pop(0) # First one is the reference image
        return similar_images


    def get_random_images(self, num: int) -> list[dict]:
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
        images = self.search_images(query, self.config.MULTIMODAL_INDEX_NAME)
        return images
    
    def import_images(self, image_path: str) -> None:
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
        print("Importing images...")

        def import_image(file):
            print(f"Found: {file}")
            file_name, extension = self.get_file_name_and_extension(file)
            if extension == 'jpg':
                extension = 'jpeg' # Fix
            if extension in self.config.IMAGE_FORMATS:
                image_base64 = self.get_image_base64(image_path + file, format=extension)
                image = self.store_image(extension, image_base64, file_name)
                return image

        imported_images = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [executor.submit(import_image, file) for file in os.listdir(image_path)]

            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                if image is not None:
                    imported_images.append(image)

        print(f"Total images imported: {len(imported_images)}")

    def get_image_catalog_count_info(self) -> int:
        return self.clients.opensearch_client.count(index=self.config.MULTIMODAL_INDEX_NAME)

    def get_text_catalog_search(self, query: dict) -> int:
        return self.clients.opensearch_client.search(body=query, index=self.config.TEXT_INDEX_NAME)

    def generate_image(self, prompt: str) -> str:
        body = json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt,
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": self.config.DEFAULT_IMAGE_HEIGHT,
                    "width": self.config.DEFAULT_IMAGE_WIDTH,
                },
            }
        )

        response = self.clients.bedrock_runtime_client.invoke_model(
            body=body,
            modelId=self.config.IMAGE_GENERATION_MODEL_UD,
        )

        response_body = json.loads(response.get("body").read())

        self.usage.update('images', 1)

        return response_body