#!/usr/bin/env python3

import argparse
import json
import os
import queue
import re
import threading
import traceback
import urllib3

from datetime import datetime
from typing import Generator

import botocore

import gradio as gr
from gradio.components.chatbot import FileMessage
from gradio.components.multimodal_textbox import MultimodalData

from rich import print


# Fix for "Error: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead."
# No other need to import numpy than for this fix
import numpy as np
np.float_ = np.float64

from libs import between_xml_tag, load_json_config
from config import Config
from clients import Clients
from utils import Utils
from tools import Tools, ToolError

# Fix to avoid the "The current process just got forked..." warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


TEXT_INDEX_CONFIG = load_json_config("./Config/text_vector_index.json")
MULTIMODAL_INDEX_CONFIG = load_json_config("./Config/multimodal_vector_index.json")
EXAMPLES = load_json_config('./Config/examples.json')


class MultimodalChat:

    def __init__(self, import_images: bool = True):
        self.config = Config()

        self.state = {
            "sketchbook": [],
            "sketchbook_current_page": 0,
            "archive": set(),
            "improvements": ""
        }

        # Create IMAGES_PATH if not exists
        os.makedirs(os.path.dirname(self.config.IMAGE_PATH), exist_ok=True)

        # Create OUTPUT_PATH if not exists
        os.makedirs(os.path.dirname(self.config.OUTPUT_PATH), exist_ok=True)

        self.output_queue = queue.Queue()

        self.clients = Clients(self.config.AWS_REGION, self.config.OPENSEARCH_HOST, self.config.OPENSEARCH_PORT)
        self.utils = Utils(self.config, self.clients)
        self.tools = Tools(self.config, self.utils, self.state, self.output_queue)

        self.utils.create_index(self.config.TEXT_INDEX_NAME, TEXT_INDEX_CONFIG)
        self.utils.create_index(self.config.MULTIMODAL_INDEX_NAME, MULTIMODAL_INDEX_CONFIG)

        if import_images:
            self.utils.import_images(self.config.IMAGE_PATH)

    def run(self):
        print("Starting the chatbot...")

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
            fn=self.chat_function,
            type="messages",
            title="Yet Another Chatbot",
            description="Your Helpful AI Assistant. I can search and browse the web, search Wikipedia, the news, and maps, run Python code that I write, write long articles, generate, download and compare images, access arXiv research papers, and transform content into an audio conversation.",
            chatbot=custom_chatbot,
            textbox=custom_textbox,
            multimodal=True,
            examples=formatted_examples,
            examples_per_page=2,
            additional_inputs=[
                gr.Checkbox(self.config.STREAMING, label="Streaming"),
                gr.Slider(0, 1, value=self.config.DEFAULT_TEMPERATURE, label="Temperature"),
                gr.Textbox(self.config.DEFAULT_SYSTEM_PROMPT, label="System Prompt"),
                gr.State(self.state),
            ],
            fill_height=True,
        )

        abs_image_path = os.path.abspath(self.config.IMAGE_PATH)
        abs_output_path = os.path.abspath(self.config.OUTPUT_PATH)
        allowed_paths = [abs_image_path, abs_output_path]
        print(f"Allowed paths: {', '.join(allowed_paths)}")

        chat_interface.launch(allowed_paths=allowed_paths)

    def process_response_message(self, response_message: dict) -> None:
        """
        Process the response message and update the state with the output content.

        Args:
            response_message (dict): The response message from the AI model.
            state (dict): The current state of the application.
            output_queue (queue.Queue): A queue to put the output into.
        """
        output_content = []
        if response_message['role'] == 'assistant' and 'content' in response_message:
            for c in response_message['content']:
                if 'text' in c:
                    cleaned_text = self.utils.remove_specific_xml_tags(c['text'])
                    
                    # Remove <img> tags that link to images not in the catalog
                    def remove_invalid_images(match):
                        img_tag = match.group(0)
                        src_match = re.search(r'src="file=(.*?)"', img_tag)
                        if src_match:
                            filename = src_match.group(1)
                            if os.path.exists(filename):
                                return img_tag
                        return ''
                    
                    cleaned_text = re.sub(r'<img.*?>', remove_invalid_images, cleaned_text)
                    
                    output_content.append(cleaned_text)

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
            self.output_queue.put({"format": "text", "text": m})

    def handle_response(self, response_message: dict) -> dict|None:
        """
        Handle the response from the AI model and process any tool use requests.

        This function takes the response message from the AI model and the current state,
        processes any tool use requests within the response, and generates follow-up
        content blocks with tool results or error messages.

        Args:
            response_message (dict): The response message from the AI model containing
                                    content blocks and potential tool use requests.
            state (dict): The current state of the chat interface.
            output_queue (queue.Queue): A queue to put the output into.

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
                    tool_result_value = self.utils.get_tool_result(tool_use_block)

                    if tool_result_value is not None:
                        follow_up_content_blocks.append({
                            "toolResult": {
                                "toolUseId": tool_use_block['toolUseId'],
                                "content": [
                                    { "json": { "result": tool_result_value } }
                                ]
                            }
                        })

                except Exception as e:
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

    def format_messages_for_bedrock_converse(self, message: dict, history: list[dict]) -> list[dict]:
        """
        Format messages for the Bedrock converse API.

        This function takes a message, conversation history, and state, and formats them
        into a structure suitable for the Bedrock Converse API. It processes text and file
        contents, handles different message types, and prepares image and document data.

        Args:
            message (dict): The latest user message.
            history (list): A list of previous messages in the conversation.
            state (dict): The current state of the application.
            output_queue (queue.Queue): A queue to put the output into.

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
                message_content.append({"text": m_text})
                append_message = True
            for file in m_files:
                file = file['path']
                file_name, extension = self.utils.get_file_name_and_extension(os.path.basename(file))
                if extension == 'jpg':
                    extension = 'jpeg' # Fix
                if extension in self.config.IMAGE_FORMATS:

                    file_content = self.utils.get_image_bytes(
                        file,
                        max_image_size=self.config.MAX_INFERENCE_IMAGE_SIZE,
                        max_image_dimension=self.config.MAX_INFERENCE_IMAGE_DIMENSIONS,
                    )

                    image_base64 = self.utils.get_image_base64(
                        file,
                        format=extension,
                        max_image_size=self.config.MAX_CHAT_IMAGE_SIZE,
                        max_image_dimension=self.config.MAX_CHAT_IMAGE_DIMENSIONS
                    )
                    image = self.utils.store_image(extension, image_base64)
                    message_content.append({
                        "text": f"The previous image has been stored in the image catalog with 'image_id': {image['id']} and this description:\n\n{image['description']}"
                    })

                    append_message = True
                elif self.config.HANDLE_DOCUMENT_TO_TEXT_IN_CODE:
                    file_name_with_extension = f'{file_name}.{extension}'
                    print(f"Importing '{file_name_with_extension}'...")
                    try:
                        if extension == 'pdf':
                            file_text = self.utils.process_pdf_document(file)
                        elif self.utils.is_text_file(file):
                                with open(file, 'r', encoding='utf-8') as f:
                                    file_text = f.read()
                        else:
                            file_text = self.utils.process_non_pdf_documents(file)
                        file_message = between_xml_tag(file_text, 'file', {'filename': file_name_with_extension})
                        self.utils.add_to_text_index(file_message, file_name_with_extension, {'filename': file_name_with_extension})
                        message_content.append({ "text": file_message })
                        message_content.append({ "text": f"File '{file_name_with_extension}' has been imported into the archive." })
                    except Exception as ex:
                        error_message = f"Error processing {file_name}.{extension} file: {ex}"
                        message_content.append({ "text": error_message })
                        print(error_message)
                        traceback.print_exc()
                elif extension in self.config.DOCUMENT_FORMATS:
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

        # Remove the last user message added at the beginning of this function
        history.pop()

        return messages

    def manage_conversation_flow(self, messages: list[dict], temperature: float, system_prompt: str) -> None:
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
            output_queue (queue.Queue): A queue to put the output into.

        Returns:
            str: The final response from the AI model, with XML tags removed.

        Note:
            This function uses global variables MAX_LOOPS and TOOLS, and relies on external functions
            invoke_text_model and handle_response.
        """
        loop_count = 0
        continue_loop = True

        # Add current date and time to the system prompt
        current_date_and_day_of_the_week = datetime.now().strftime("%a, %Y-%m-%d")
        current_time = datetime.now().strftime("%I:%M:%S %p")
        system_prompt_with_improvements = system_prompt + f"\nKeep in mind that today is {current_date_and_day_of_the_week} and the current time is {current_time}."

        if len(self.state['improvements']) > 0:
            system_prompt_with_improvements += "\n\nImprovements:\n" + self.state['improvements']

        while continue_loop:

            response = self.utils.invoke_text_model(messages, system_prompt_with_improvements, temperature, tools=self.tools.tools_json)

            response_message = response['output']['message']
            messages.append(response_message)

            self.process_response_message(response_message)

            follow_up_message = self.handle_response(response_message)

            if follow_up_message is None:
                # No remaining work to do, return final response to user
                continue_loop = False
            else:
                loop_count = loop_count + 1
                if loop_count >= self.config.MAX_LOOPS:
                    print(f"Hit loop limit: {loop_count}")
                    continue_loop = False
                else:
                    messages.append(follow_up_message)

    def process_streaming_chunk(self, chunk: dict, messages: list[dict], stream_state: dict) -> None:
        match chunk:
            case {'messageStart': message_start}:
                stream_state['tool_use_block'] = None
                stream_state['new_content'] = ''
            case {'messageStop': message_stop}:
                pass
            case {'metadata': metadata}:
                print(f"Metadata: {metadata}")
                for metric, value in metadata['usage'].items():
                    self.utils.usage.update(metric, value)
                print(self.utils.usage)
            case {'contentBlockStart': content_block_start}:
                match content_block_start['start']:
                    case {'toolUse': tool_use_start}:
                        print(f"ToolUse: {tool_use_start}")
                        stream_state['tool_use_block'] = tool_use_start
                        stream_state['tool_use_block']['input'] = ''
                    case _:
                        print(f"Unknown start: {content_block_start['start']}")
            case {'contentBlockDelta': content_block_delta}:
                match content_block_delta['delta']:
                    case {'text': text}:
                        stream_state['new_content'] += text
                        self.output_queue.put({"format": "text", "text": text})
                    case {'toolUse': tool_use_delta}:
                        stream_state['tool_use_block']['input'] += tool_use_delta['input']
                    case _:
                        print(f"Unknown delta: {chunk}")
            case {'contentBlockStop': content_block_stop}:
                if len(stream_state['new_content']) > 0:
                    stream_state['new_message'] = {
                        "role": "assistant",
                        "content": [ { "text": stream_state['new_content'] } ],
                    }
                    stream_state['new_content'] = ''
                elif stream_state['new_message'] is None:
                    stream_state['new_message'] = {
                        "role": "assistant",
                        "content": [],
                    }
                if stream_state['tool_use_block']:
                    try:
                        # Fix the input to be a dictionary
                        stream_state['tool_use_block']['input'] = json.loads(stream_state['tool_use_block']['input'])
                        stream_state['new_message']['content'].append( { "toolUse": stream_state['tool_use_block'].copy() } )

                        tool_result_value = self.tools.get_tool_result(stream_state['tool_use_block'])

                        if tool_result_value is not None:
                            stream_state['follow_up_content_blocks'].append({
                                "toolResult": {
                                    "toolUseId": stream_state['tool_use_block']['toolUseId'],
                                    "content": [
                                        { "json": { "result": tool_result_value } }
                                    ]
                                }
                            })

                    except ToolError as e:
                        print(f"Error processing tool use: {e}")
                        stream_state['follow_up_content_blocks'].append({
                            "toolResult": {
                                "toolUseId": stream_state['tool_use_block']['toolUseId'],
                                "content": [ { "text": repr(e) } ],
                                "status": "error"
                            }
                        })

                    finally:
                        messages.append(stream_state['new_message'])
                        stream_state['tool_use_block'] = None

            case _:
                print(f"Unknown streaming chunk: {chunk}")

    def manage_conversation_flow_stream(self, messages: list[dict], temperature: float, system_prompt: str) -> None:
        """
        Run a conversation loop with the AI model using streaming, processing responses and handling tool usage.

        This function manages the conversation flow between the user and the AI model using a streaming approach.
        It sends messages to the model, processes the responses in real-time, handles any tool usage requests,
        and continues the conversation until a final response is ready or the maximum number of loops is reached.

        Args:
            messages (list): A list of message dictionaries representing the conversation history.
            system_prompt (str): The system prompt to guide the AI model's behavior.
            temperature (float): The temperature parameter for the AI model's response generation.
            state (dict): The current state of the chat interface.
            output_queue (queue.Queue): A queue to put the output into.

        Returns:
            None

        Note:
            - This function uses global variables MAX_LOOPS, TOOLS, and bedrock_runtime_client.
            - It handles streaming responses from the AI model and processes them in chunks.
            - Tool usage is handled in real-time as the response is being generated.
        """
        global bedrock_runtime_client

        continue_loop = True

        # Add current date and time to the system prompt
        current_date_and_day_of_the_week = datetime.now().strftime("%a, %Y-%m-%d")
        current_time = datetime.now().strftime("%I:%M:%S %p")
        system_prompt_with_improvements = system_prompt + f"\nKeep in mind that today is {current_date_and_day_of_the_week} and the current time is {current_time}."

        if len(self.state['improvements']) > 0:
            system_prompt_with_improvements += "\n\nImprovements:\n" + self.state['improvements']

        while continue_loop:

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

            if self.tools.tools_json:
                converse_body["toolConfig"] = {"tools": self.tools.tools_json}

            print("Thinking...")

            retry_flag = True
            while retry_flag:

                streaming_response = self.clients.bedrock_runtime_client.converse_stream(**converse_body)

                stream_state = {
                    'new_content': '',
                    'new_message': None,
                    'tool_use_block': None,
                    'follow_up_content_blocks': [],
                }

                try:
                    for chunk in streaming_response['stream']:
                        self.process_streaming_chunk(chunk, messages, stream_state)
                except (urllib3.exceptions.ReadTimeoutError, botocore.exceptions.EventStreamError) as e:
                    print("Retrying...")
                    continue

                retry_flag = False

            if len(stream_state['follow_up_content_blocks']) > 0:
                follow_up_message = {
                    "role": "user",
                    "content": stream_state['follow_up_content_blocks'],
                }
                messages.append(follow_up_message)
            else:
                continue_loop = False

    def chat_function(self, message: dict, history: list[dict], streaming: bool, temperature: float, system_prompt: str, state: dict) -> Generator[str, None, None]:
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
        def format_response(response):
            response = self.utils.process_image_placeholders(response)
            response = self.utils.remove_specific_xml_tags(response)
            response = self.utils.replace_specific_xml_tags(response)
            return response

        if message['text'] == '':
            yield "Please enter a message."
        else:
            messages = self.format_messages_for_bedrock_converse(message, history)
            if streaming:
                target_function = self.manage_conversation_flow_stream
            else:
                target_function = self.manage_conversation_flow
            thread = threading.Thread(target=target_function, args=(messages, temperature, system_prompt))
            thread.start()

            num_dots = 0
            tot_dots = 3
            response = ""
            while thread.is_alive() or not self.output_queue.empty():
                try:
                    output = self.output_queue.get(timeout=0.5)
                    response += output['text']
                    if not streaming:
                        response += "\n"
                    yield format_response(response) # Yield the response as it's being generated
                    self.output_queue.task_done()
                except queue.Empty:
                    if len(response) > 0:
                        num_dots = (num_dots % tot_dots) + 1
                        yield f"{format_response(response)}\n{'.' * num_dots}"
                    continue  # If the queue is empty, continue the loop

            yield format_response(response) # To avoid the last dot(s)

            print()
            if not response:
                yield "No response generated."

    def reset_index(self):
        self.utils.delete_index(self.config.TEXT_INDEX_NAME)
        self.utils.delete_index(self.config.MULTIMODAL_INDEX_NAME)


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


    if args.reset_index:
        multimodal_chat = MultimodalChat(import_images=False)
        multimodal_chat.reset_index()
        print("Indexes reset.")
        exit()

    multimodal_chat = MultimodalChat()
    multimodal_chat.run()
