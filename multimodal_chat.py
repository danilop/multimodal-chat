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

    def __init__(self, import_images: bool = True) -> None:
        """
        Initialize the MultimodalChat class.

        Args:
            import_images (bool, optional): Whether to import images on initialization. Defaults to True.
        """
        self.config = Config()

        # Create IMAGES_PATH if not exists
        os.makedirs(os.path.dirname(self.config.IMAGE_PATH), exist_ok=True)

        # Create OUTPUT_PATH if not exists
        os.makedirs(os.path.dirname(self.config.OUTPUT_PATH), exist_ok=True)

        self.clients = Clients(self.config)
        self.utils = Utils(self.config, self.clients)

        self.utils.create_index(self.config.TEXT_INDEX_NAME, TEXT_INDEX_CONFIG)
        self.utils.create_index(self.config.MULTIMODAL_INDEX_NAME, MULTIMODAL_INDEX_CONFIG)

        if import_images:
            self.utils.import_images(self.config.IMAGE_PATH)

    def reset_state(self, state: dict) -> None:
        state['messages'] = []
        state['sketchbook'] = {}
        state['sketchbook_current_section'] = {}
        state['checklist'] = {}
        state['archive'] = set()
        state['documents'] = {}
        state['improvements'] = ""
        state['files'] = {}

    def run(self) -> None:
        """
        Start the chatbot interface using Gradio.

        This method sets up the Gradio interface for the chatbot, including the chat input,
        file upload, and example selection functionality.
        """
        print("Starting the chatbot...")

        # Formatted for type "messages" (multimodal)
        formatted_examples = [{"text": example} for example in EXAMPLES]

        with open("update.js", "r") as f:
            head_script = '<script>' + f.read() + '</script>'

        with gr.Blocks(title="Multimodal Chat", fill_height=True, fill_width=True, head=head_script) as app:

            old_content = {}

            state = gr.State({})

            self.reset_state(state.value)

            gr.Markdown(
                """
                # Yet Another Intelligent Assistant (YAIA)
                A multimodal chat interface with access to many tools. To learn more, start with the examples.
                """,
                elem_id="header-title",
            )
            
            with gr.Row(elem_id="chatbotrow", equal_height=True):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    show_label=False,
                    type="messages",
                    examples=formatted_examples,
                    show_copy_button=True,
                    show_copy_all_button=True,
                )

                tools = Tools(self.config, state.value, self.utils)

                with gr.Column(visible=False, elem_id="side-column") as side_column:
                    @gr.render(inputs=[state])
                    def update_tabs(state_value):
                        for id in state_value['sketchbook']:
                            formatted_id = self.utils.format_string(id)
                            sketchbook = state_value['sketchbook'][id]
                            content = tools.render_sketchbook(sketchbook, forPreview=True) 
                            old_content[id] = content
                            with gr.Tab(formatted_id):
                                gr.Markdown(
                                    content,
                                    elem_classes="tab-content",
                                    line_breaks=True,
                                    show_copy_button=True,
                                )
                        for filename in state_value['files']:
                            code_fence_language = state_value['files'][filename]['code_fence_language']
                            content = "```" + code_fence_language + "\n" + state_value['files'][filename]['content'] + "\n```"
                            with gr.Tab(f"File: {filename}"):
                                gr.Markdown(
                                    content,
                                    elem_classes="tab-content",
                                    line_breaks=True,
                                )

            def change_visibility(state):
                if len(state['sketchbook']) > 0 or len(state['files']) > 0:
                    return gr.Column(visible=True)
                else:
                    return gr.Column(visible=False)

            state.change(change_visibility, inputs=[state], outputs=[side_column])

            chat_input = gr.MultimodalTextbox(
                elem_id="chat-input",
                show_label=False,
                placeholder="Enter your instructions and press enter.",
                file_count='multiple',
                autofocus=True,
            )

            def add_message(history: list, message):
                for x in message["files"]:
                    history.append({"role": "user", "content": {"path": x}})
                if message["text"] is not None:
                    history.append({"role": "user", "content": message["text"]})
                return history, ''

            chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input], queue=False).then(
                self.chat_function, inputs=[chatbot, state], outputs=[chatbot, state]
            )

            def add_example(evt: gr.SelectData):
                return evt.value
            
            chatbot.example_select(add_example, None, chat_input)

            def handle_undo(history, undo_data: gr.UndoData):
                return history[:undo_data.index], history[undo_data.index]['content']

            chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])

            def handle_retry(history, state, retry_data: gr.RetryData):
                yield from self.chat_function(history[:retry_data.index + 1], state)

            chatbot.retry(handle_retry, [chatbot, state], [chatbot, state])

        abs_image_path = os.path.abspath(self.config.IMAGE_PATH)
        abs_output_path = os.path.abspath(self.config.OUTPUT_PATH)
        allowed_paths = [abs_image_path, abs_output_path]
        print(f"Allowed paths: {', '.join(allowed_paths)}")

        app.launch(allowed_paths=allowed_paths)

    def process_response_message(self, output_queue: queue.Queue, response_message: dict) -> None:
        """
        Process the response message and update the state with the output content.

        Args:
            response_message (dict): The response message from the AI model.
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
            output_queue.put({"format": "text", "text": m})

    def handle_response(self, output_queue: queue.Queue, response_message: dict) -> dict|None:
        """
        Handle the response from the AI model and process any tool use requests.

        Args:
            response_message (dict): The response message from the AI model containing
                                    content blocks and potential tool use requests.

        Returns:
            dict or None: A follow-up message containing tool results if any tools were used,
                        or None if no tools were used.
        """

        response_content_blocks = response_message['content']
        follow_up_content_blocks = []

        for content_block in response_content_blocks:
            if 'toolUse' in content_block:
                tool_use_block = content_block['toolUse']

                try:
                    tool_result_value, formatted_tool_use_name, tool_metadata = self.utils.get_tool_result(tool_use_block)

                    if len(tool_metadata) > 0:
                        output_queue.put({"format": "metadata", "title": formatted_tool_use_name, "metadata": tool_metadata})

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

    def format_new_messages_for_bedrock_converse(self, history: list[dict], state: dict) -> list[dict]:
        """
        Format messages for the Bedrock converse API.

        Args:
            history (list[dict]): A list of previous messages in the conversation.

        Returns:
            list[dict]: A list of formatted messages ready for the Bedrock converse API.

        Note:
            This function handles various types of content including text, images, and documents.
            It also processes file uploads and stores images
        """

        messages = []
        message_content = []

        skip_next = False
        for m in history:
            append_message = False
            m_role = m['role']
            m_content = m['content']

            # To skip metadata messages from tools
            if 'metadata' in m and 'title' in m['metadata'] and m['metadata']['title'] is not None:
                continue

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
                if 'path' in file: # Remove?
                    file = file['path'] # Remove?
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
                    try:
                        if file_name_with_extension in state['documents']:
                            print(f"File '{file_name_with_extension}' already imported.")
                            file_text = state['documents'][file_name_with_extension]
                        else:
                            print(f"Importing '{file_name_with_extension}'...")
                            if extension == 'pdf':
                                file_text = self.utils.process_pdf_document(file)
                            elif extension == 'pptx':
                                file_text = self.utils.process_pptx_document(file)
                            elif self.utils.is_text_file(file):
                                with open(file, 'r', encoding='utf-8') as f:
                                    file_text = f.read()
                            else:
                                file_text = self.utils.process_other_document_formats(file)
                            state['documents'][file_name_with_extension] = file_text
                        file_message = between_xml_tag(file_text, 'file', {'filename': file_name_with_extension})
                        self.utils.add_to_text_index(file_message, file_name_with_extension, {'filename': file_name_with_extension})
                        message_content.append({ "text": file_message })
                        message_content.append({ "text": f"File '{file_name_with_extension}' has been imported into the archive." })
                    except Exception as ex:
                        error_message = f"Error processing {file_name}.{extension} file: {ex}"
                        message_content.append({ "text": error_message })
                        print(error_message)
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

        return messages

    def get_system_prompt(self, system_prompt: str, state: dict) -> str:
        # Add current date and time to the system prompt
        current_date_and_day_of_the_week = datetime.now().strftime("%a, %Y-%m-%d")
        current_time = datetime.now().strftime("%I:%M:%S %p")
        full_system_prompt = f"{system_prompt}\nKeep in mind that today is {current_date_and_day_of_the_week} and the current time is {current_time}."
        if len(state['improvements']) > 0:
            full_system_prompt += "\n\nImprovements:\n" + state['improvements']

        return full_system_prompt

    def manage_conversation_flow(self, output_queue: queue.Queue, messages: list[dict], tools: Tools, temperature: float, system_prompt: str) -> None:
        """
        Run a conversation loop with the AI model, processing responses and handling tool usage.

        Args:
            messages (list[dict]): A list of message dictionaries representing the conversation history.
            temperature (float): The temperature parameter for the AI model's response generation.
            system_prompt (str): The system prompt to guide the AI model's behavior.

        Note:
            This function uses global variables MAX_LOOPS and TOOLS, and relies on external functions
            invoke_text_model and handle_response.
        """
        loop_count = 0
        continue_loop = True

        full_system_prompt = self.get_system_prompt(system_prompt, tools.state)

        while continue_loop:

            response = self.utils.invoke_text_model(messages, full_system_prompt, temperature, tools=tools.tools_json)

            response_message = response['output']['message']
            messages.append(response_message)

            self.process_response_message(output_queue, response_message)

            follow_up_message = self.handle_response(output_queue, response_message)

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

    def process_streaming_chunk(self, chunk: dict, messages: list[dict], tools: Tools, stream_state: dict) -> None:
        """
        Process a chunk of streaming data from the AI model.

        Args:
            chunk (dict): A chunk of streaming data from the AI model.
            messages (list[dict]): A list of message dictionaries representing the conversation history.
            stream_state (dict): A dictionary to store the state of the streaming process.

        Note:
            This function handles different types of content blocks and updates the conversation history
            accordingly. It also handles tool usage and updates the usage metrics.
        """
        match chunk:
            case {'messageStart': message_start}:
                stream_state['tool_use_block'] = None
                stream_state['new_content'] = ''
            case {'messageStop': message_stop}:
                messages.append(stream_state['new_message'])
            case {'metadata': metadata}:
                for metric, value in metadata['usage'].items():
                    self.utils.usage.update(metric, value)
                print(self.utils.usage)
            case {'contentBlockStart': content_block_start}:
                match content_block_start['start']:
                    case {'toolUse': tool_use_start}:
                        tool_name = tool_use_start.get('name', 'tool name missing')
                        print(f"Using tool {tool_name}...")
                        stream_state['tool_use_block'] = tool_use_start
                        stream_state['tool_use_block']['input'] = ''
                    case _:
                        print(f"Unknown start: {content_block_start['start']}")
            case {'contentBlockDelta': content_block_delta}:
                match content_block_delta['delta']:
                    case {'text': text}:
                        stream_state['new_content'] += text
                        stream_state['output_queue'].put({"format": "text", "text": text})
                    case {'toolUse': tool_use_delta}:
                        stream_state['tool_use_block']['input'] += tool_use_delta['input']
                    case _:
                        print(f"Unknown delta: {chunk}")
            case {'contentBlockStop': content_block_stop}:
                if len(stream_state['new_content'].strip().strip(' \n')) > 0:
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
                    if stream_state['tool_use_block']['input'] == '':
                        stream_state['tool_use_block']['input'] = {}  
                    else:
                        try:
                            # Load the tool input JSON string to a dictionary
                            stream_state['tool_use_block']['input'] = json.loads(stream_state['tool_use_block']['input'])
                        except json.JSONDecodeError:
                            pass

                    stream_state['new_message']['content'].append( { "toolUse": stream_state['tool_use_block'].copy() } )

                    try:
                        tool_result_value, formatted_tool_use_name, tool_metadata = tools.get_tool_result(stream_state['tool_use_block'])

                        if len(tool_metadata) > 0:
                            stream_state['output_queue'].put({"format": "metadata", "title": formatted_tool_use_name, "metadata": tool_metadata})

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
                        print(traceback.format_exc())
                        stream_state['follow_up_content_blocks'].append({
                            "toolResult": {
                                "toolUseId": stream_state['tool_use_block']['toolUseId'],
                                "content": [ { "text": repr(e) } ],
                                "status": "error"
                            }
                        })

                    finally:
                        stream_state['tool_use_block'] = None

            case _:
                print(f"Unknown streaming chunk: {chunk}")

    def manage_conversation_flow_stream(self, output_queue: queue.Queue, messages: list[dict], tools: Tools, temperature: float, system_prompt: str) -> None:
        """
        Run a conversation loop with the AI model using streaming, processing responses and handling tool usage.

        Args:
            messages (list[dict]): A list of message dictionaries representing the conversation history.
            temperature (float): The temperature parameter for the AI model's response generation.
            system_prompt (str): The system prompt to guide the AI model's behavior.

        Note:
            This function handles streaming responses from the AI model and processes them in chunks.
            Tool usage is handled in real-time as the response is being generated.
        """
        continue_loop = True

        full_system_prompt = self.get_system_prompt(system_prompt, tools.state)

        while continue_loop:

            converse_body = {
                "modelId": self.config.TEXT_MODEL,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": self.config.MAX_TOKENS,
                    "temperature": temperature,
                    },
            }

            if system_prompt is not None:
                converse_body["system"] = [{"text": system_prompt}]

            if tools:
                converse_body["toolConfig"] = {"tools": tools.tools_json}

            print("Thinking...")

            retry_flag = True
            while retry_flag:

                streaming_response = self.clients.bedrock_runtime_client_text_model.converse_stream(**converse_body)

                stream_state = {
                    'output_queue': output_queue,
                    'new_content': '',
                    'new_message': None,
                    'tool_use_block': None,
                    'follow_up_content_blocks': [],
                }

                try:
                    for chunk in streaming_response['stream']:
                        self.process_streaming_chunk(chunk, messages, tools, stream_state)
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

    def chat_function(self, history: list[dict], state: dict) -> Generator[list[dict], None, None]:
        """
        Process a chat message and generate a response using an AI model.

        Args:
            history (list[dict]): A list of previous messages in the conversation.

        Yields:
            list[dict]: The updated conversation history including the generated response.

        Note:
            This function modifies the 'state' dictionary to store output for display in the chat interface.
            It handles both text and file inputs, and can display generated images in the response.
        """
        def format_response(text):
            cleaned_text = text
            cleaned_text = self.utils.remove_specific_xml_tags(cleaned_text)
            cleaned_text = self.utils.replace_specific_xml_tags(cleaned_text)
            return cleaned_text

        def find_images_and_files_in_response(response):
            try:
                results = self.utils.process_image_and_file_placeholders_for_chat(response)
            except Exception as e:
                error_message = f"Error processing image placeholders: {e}"
                print(error_message)
                return [{"format": "text", "text": error_message}]
            return results

        tools = Tools(self.config, state, self.utils)
        output_queue = queue.Queue()

        # Last user messages, can be more than one if files are uploaded
        new_history = []
        for i in range(len(history)-1, -1, -1):
            if history[i]['role'] == 'assistant':
                break
            if history[i]['role'] == 'user':
                new_history.insert(0, history[i])

        if len(new_history) == len(history):
            self.reset_state(state)

        print(f"State: {state}")

        new_messages = self.format_new_messages_for_bedrock_converse(new_history, tools.state)

        messages = state['messages']
        messages.extend(new_messages)

        if self.config.STREAMING:
            target_function = self.manage_conversation_flow_stream
        else:
            target_function = self.manage_conversation_flow
        thread = threading.Thread(target=target_function, args=(output_queue, messages, tools, self.config.TEMPERATURE, self.config.SYSTEM_PROMPT))
        thread.start()

        num_dots = 0
        tot_dots = 3
        history.append({"role": "assistant", "content": ""})

        while thread.is_alive() or not output_queue.empty():
            try:
                output = output_queue.get(timeout=0.5)
                match output['format']:
                    case 'text':
                        history[-1]['content'] += output['text']
                        history[-1]['content'] = format_response(history[-1]['content'])
                        results = find_images_and_files_in_response(history[-1]['content'])
                        if len(results) > 1:
                            history[-1]['content'] = ''
                            for item in results:
                                match item['format']:
                                    case 'text':
                                        history[-1]['content'] += item['text']
                                    case 'file':
                                        if len(history[-1]['content']) == 0:
                                            history.pop()
                                        content = {"path": item['filename'], "alt_text": item['description']}
                                        history.append({"role": "assistant", "content": content})
                                        history.append({"role": "assistant", "content": ""})
                    case 'metadata':
                        title = output['title']
                        metadata = output['metadata']
                        if not metadata.startswith("```"):
                            metadata = f"```\n{metadata}\n```"
                        if len(history[-1]['content']) == 0:
                            history.pop()
                        history.append({"role": "assistant", "content": metadata, "metadata": { "title": title }})
                        history.append({"role": "assistant", "content": ""})
                    case _:
                        print(f"Unknown output format: {output['format']}")
                yield history, state
                output_queue.task_done()
            except queue.Empty:
                num_dots = (num_dots % tot_dots) + 1
                if type(history[-1]['content']) == str:
                    old_content = history[-1]['content']
                    history[-1]['content'] += f"\n{'.' * num_dots}"
                    yield history, state
                    history[-1]['content'] = old_content
                else:
                    history.append({"role": "assistant", "content": f"\n{'.' * num_dots}"})
                    yield history, state
                    history.pop()
                continue

        # To clean up dots in the last message and avoid empty messages
        if len(history[-1]['content']) == 0:
            history.pop()

        state['messages'] = messages

        yield history, state


    def reset_index(self) -> None:
        """
        Reset the text and multimodal indexes.

        This method deletes and recreates the text and multimodal indexes in the database.
        """
        self.utils.delete_index(self.config.TEXT_INDEX_NAME)
        self.utils.delete_index(self.config.MULTIMODAL_INDEX_NAME)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the chatbot application.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Note:
        This function sets up an argument parser to handle the --reset-index flag.
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
