import concurrent.futures
import io
import json
import os
import queue
import random
import re
import tempfile
import time
import pypandoc
import urllib
import uuid

from datetime import datetime
from urllib.parse import urlparse

import arxiv
import wikipedia

from duckduckgo_search import DDGS
from selenium import webdriver
from selenium.webdriver.common.by import By
from pydub import AudioSegment

from libs import load_json_config, mark_down_formatting, between_xml_tag

from config import Config
from utils import Utils, ImageNotFoundError


class ToolError(Exception):
    """
    Custom exception class for tool-related errors.

    This exception is raised when there's an issue with tool execution or usage
    in the chat system. It can be used to handle specific errors related to
    tools and provide meaningful error messages to the user or the system.
    """
    pass


class Tools:
    """
    A class to manage and execute tools for the chatbot.

    This class loads tool configurations from a JSON file, initializes utility functions,
    and provides methods to execute various tools based on the tool name and input.
    """

    def __init__(self, config: Config, utils: Utils) -> None:
        """
        Initialize the Tools class with the given configuration, utilities, state, and output queue.

        Args:
            config (Config): The configuration object.
            utils (Utils): The utility object.
            state (dict): The state of the chatbot.
        """

        self.config = config
        self.utils = utils

        self.tools_json = load_json_config('./Config/tools.json')

        self.tool_functions = {
            'python': self.get_tool_result_python,
            'duckduckgo_text_search': self.get_tool_result_duckduckgo_text_search,
            'duckduckgo_news_search': self.get_tool_result_duckduckgo_news_search,
            'duckduckgo_maps_search': self.get_tool_result_duckduckgo_maps_search,
            'duckduckgo_images_search': self.get_tool_result_duckduckgo_images_search,
            'wikipedia_search': self.get_tool_result_wikipedia_search,
            'wikipedia_geodata_search': self.get_tool_result_wikipedia_geodata_search,
            'wikipedia_page': self.get_tool_result_wikipedia_page,
            'browser': self.get_tool_result_browser,
            'retrive_from_archive': self.get_tool_result_retrive_from_archive,
            'store_in_archive': self.get_tool_result_store_in_archive,
            'sketchbook': self.get_tool_result_sketchbook,
            'checklist': self.get_tool_result_checklist,
            'generate_image': self.get_tool_result_generate_image,
            'search_image_catalog': self.get_tool_result_search_image_catalog,
            'similarity_image_catalog': self.get_tool_result_similarity_image_catalog,
            'random_images': self.get_tool_result_random_images,
            'get_image_by_id': self.get_tool_result_get_image_by_id,
            'image_catalog_count': self.get_tool_result_image_catalog_count,
            'download_image_into_catalog': self.get_tool_result_download_image_into_catalog,
            'personal_improvement': self.get_tool_result_personal_improvement,
            'arxiv': self.get_tool_result_arxiv,
            'save_text_file': self.get_tool_result_save_text_file,
            'check_if_file_exists': self.get_tool_result_check_if_file_exists,
            'conversation': self.get_tool_result_conversation,
        }

        self.check_tools_consistency()

    def set_state(self, state: dict) -> None:
        self.state = state

    def get_tool_result_python(self, tool_input: dict) -> str:
        """
        Execute a Python script using AWS Lambda and process the result.

        Args:
            tool_input (dict): A dictionary containing the 'script' key with the Python code to execute.

        Returns:
            str: The output of the Python script execution, wrapped in XML tags.

        Note:
            - The function uses the AWS_LAMBDA_FUNCTION_NAME from the config for the Lambda function name.
            - It adds the script and its output to the chat interface's state for display.
            - The output is truncated if it exceeds MAX_OUTPUT_LENGTH from the config.
            - If images are generated during script execution, they are stored in the image catalog.
        """
        input_script = tool_input["script"]
        install_modules = tool_input.get("install_modules", [])
        number_of_images = tool_input.get("number_of_images", 0)

        if type(install_modules) == str:
            try:
                install_modules = json.loads(install_modules)
            except Exception as e:
                error_message = f"Error: {e}"
                print(error_message)
                return error_message

        print(f"Script:\n{input_script}")
        print(f"Install modules: {install_modules}")
        print(f"Number of images: {number_of_images}")

        start_time = time.time()
        event = {"input_script": input_script, "install_modules": install_modules}

        print("Invoking Lambda function...")
        result = self.utils.invoke_lambda_function(self.config.AWS_LAMBDA_FUNCTION_NAME, event) 
        output = result.get("output", "")
        images = result.get("images", [])

        end_time = time.time()
        elapsed_time = end_time - start_time
        len_output = len(output)
        print(f"Output length: {len_output}")
        print(f"Images: {len(images)}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        if len_output == 0:
            warning_message = "No output printed."
            print(warning_message)
            return warning_message
        
        if len_output > self.config.MAX_OUTPUT_LENGTH:
            output = output[:self.config.MAX_OUTPUT_LENGTH] + "\n... (truncated)"

        print(f"Output:\n---\n{output}\n---.")

        if len(images) != number_of_images:
            warning_message = f"Expected {number_of_images} images, but found {len(images)}"
            print(warning_message)
            return f"{output}\n\n{warning_message}"
        
        for i in images:
            # Extract the image format from the file extension
            image_path = i['path']
            image_format = os.path.splitext(image_path)[1][1:] # Remove the leading dot
            image_format = 'jpeg' if image_format == 'jpg' else image_format # Quick fix
            image_base64 = self.utils.get_image_base64(i['base64'],
                                            format=image_format,
                                            max_image_size=self.config.MAX_CHAT_IMAGE_SIZE,
                                            max_image_dimension=self.config.MAX_CHAT_IMAGE_DIMENSIONS)

            image = self.utils.store_image(image_format, image_base64)
            output += f"\nImage {image_path} has been stored in the image catalog with 'image_id': {image['id']}"

        return f"{between_xml_tag(output, 'output')}"

    def get_tool_result_duckduckgo_text_search(self, tool_input: dict) -> str:
        """
        Perform a DuckDuckGo text search and store the results in the archive.

        Args:
            tool_input (dict): A dictionary containing the 'keywords' for the search.

        Returns:
            str: XML-tagged output containing the search results.

        Note:
            This function uses MAX_SEARCH_RESULTS from the config to limit the number of results.
            It also adds the search results to the text index for future retrieval.
        """
        search_keywords = tool_input["keywords"]
        print(f"Keywords: {search_keywords}")
        try:
            results = DDGS().text(search_keywords, max_results=self.config. MAX_SEARCH_RESULTS)
            output = json.dumps(results)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_duckduckgo_news_search(self, tool_input: dict) -> str:
        """
        Perform a DuckDuckGo news search and store the results in the archive.

        Args:
            tool_input (dict): A dictionary containing the 'keywords' for the search.

        Returns:
            str: XML-tagged output containing the search results.

        Note:
            This function uses the global MAX_SEARCH_RESULTS to limit the number of results.
            It also adds the search results to the text index for future retrieval.
        """
        search_keywords = tool_input["keywords"]
        print(f"Keywords: {search_keywords}")
        try:
            results = DDGS().news(search_keywords, max_results=self.config.MAX_SEARCH_RESULTS)
            output = json.dumps(results)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_duckduckgo_maps_search(self, tool_input: dict) -> str:
        """
        Perform a DuckDuckGo maps search and store the results in the archive.

        Args:
            tool_input (dict): A dictionary containing the 'keywords' and 'place' for the search.

        Returns:
            str: XML-tagged output containing the search results.

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
                search_keywords, search_place, max_results=self.config.MAX_SEARCH_RESULTS
            )
            output = json.dumps(results)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_wikipedia_search(self, tool_input: dict) -> str:
        """
        Perform a Wikipedia search and return the results.

        Args:
            tool_input (dict): A dictionary containing the 'query' for the search.

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
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output: {output}")
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_duckduckgo_images_search(self, tool_input: dict) -> str:
        """
        Perform a DuckDuckGo images search.

        Args:
            tool_input (dict): A dictionary containing the 'keywords' for the search.

        Returns:
            str: XML-tagged output containing the search results.

        Note:
            This function uses MAX_SEARCH_RESULTS from the config to limit the number of results.
            It also adds the search results to the text index for future retrieval.
        """
        search_keywords = tool_input["keywords"]
        print(f"Keywords: {search_keywords}")
        try:
            results = DDGS().images(search_keywords, license_image='Share', max_results=self.config.MAX_SEARCH_RESULTS)
            output = json.dumps(results)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_wikipedia_geodata_search(self, tool_input: dict) -> str:
        """
        Perform a Wikipedia geosearch and return the results.

        Args:
            tool_input (dict): A dictionary containing the search parameters:
                - latitude (float): The latitude of the search center.
                - longitude (float): The longitude of the search center.
                - title (str, optional): The title of a page to search for.
                - radius (int, optional): The search radius in meters.

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
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        output = output.strip()
        print(f"Output: {output}")
        print(f"Output length: {len(output)}")

        return between_xml_tag(output, "output")

    def get_tool_result_wikipedia_page(self, tool_input: dict) -> str:
        """
        Retrieve and process a Wikipedia page, storing its content in the archive.

        This function fetches a Wikipedia page based on the given title, converts its HTML content
        to Markdown format, and stores it in the text index for future retrieval.

        Args:
            tool_input (dict): A dictionary containing the 'title' and 'keywords' keys with the Wikipedia page title and keywords for the search.
        Returns:
            str: A message indicating that the page content has been stored in the archive.

        Note:
            This function uses the wikipedia library to fetch page content and the mark_down_formatting
            function to convert HTML to Markdown. It also uses add_to_text_index to store the content
            in the archive with appropriate metadata.
        """
        search_title = tool_input.get("title", "")
        keywords = tool_input.get("keywords", "")
        print(f"Title: {search_title}")
        print(f"Keywords: {keywords}")
        try:
            page = wikipedia.page(title=search_title, auto_suggest=False)
            page_text = mark_down_formatting(page.html(), page.url)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            return error_message
        page_text = page_text.strip()
        print(f"Output length: {len(page_text)}")
        current_date = datetime.now().strftime("%Y-%m-%d")
        metadata = {"wikipedia_page": search_title, "date": current_date}
        metadata_delete = {"wikipedia_page": search_title}
        self.utils.add_to_text_index(page_text, search_title, metadata, metadata_delete)

        summary = self.retrieve_from_archive(search_title)
        print(f"Summary length: {len(summary)}")

        keywords_content = self.retrieve_from_archive(keywords)
        print(f"Keywords content length: {len(keywords_content)}")

        output = f"""The full content of the page ({len(page_text)} characters) has been stored in the archive.
            Retrieve more information from the archive using keywords or browse links to get more information.
            Here is a summary of the page:
            {between_xml_tag(summary, 'summary')}
            Additional information based on your keywords:
            {between_xml_tag(keywords_content, 'info')}"""
        
        print(f"Output length: {len(output)}")

        return output

    def get_tool_result_browser(self, tool_input: dict) -> str:
        """
        Retrieve and process content from a given URL using Selenium.

        This function uses Selenium WebDriver to navigate to the specified URL,
        retrieve the page content, convert it to Markdown format, and store it
        in the text index for future retrieval.

        Args:
            tool_input (dict): A dictionary containing the 'url' key with the target URL.
        Returns:
            str: A message indicating that the content has been stored in the archive.

        Note:
            This function uses Selenium with Chrome in headless mode to retrieve page content.
            It also uses mark_down_formatting to convert HTML to Markdown and add_to_text_index
            to store the content in the archive with appropriate metadata.
        """
        url = tool_input.get("url", "")
        keywords = tool_input.get("keywords", "")
        print(f"URL: {url}")
        print(f"Keywords: {keywords}")
        
        parsed_url = urlparse(url)
        url_file_extension = os.path.splitext(parsed_url.path)[1].lower().lstrip('.')

        if url_file_extension in self.config.DOCUMENT_FORMATS:
            # Handle document formats
            print(f"Downloading and processing document: {url}")
                    
            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                print(f"Document size: {len(content)}")

                content_as_io_bytes = io.BytesIO(content)

                if url_file_extension == 'pdf':
                    page_text = self.utils.process_pdf_document(content_as_io_bytes)
                elif url_file_extension == 'pptx':
                    page_text = self.utils.process_pptx_document(content_as_io_bytes)
                else:
                    raise Exception(f"Unsupported document format for browser: {url_file_extension}")
                
                title = None

            except Exception as e:
                error_message = f"Error downloading or processing the document: {str(e)}"
                print(error_message)
                return error_message

        else:

            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--incognito")
            options.add_argument("--window-size=1920,1080")
            
            with webdriver.Chrome(options=options) as driver:
                driver.get(url)

                title = driver.title
                print(f"Title:", title)

                page = driver.page_source
                print(f"Page length:", len(page))
                
                page_text = mark_down_formatting(page, url)
                print(f"Markdown text length:", len(page_text))

        if len(page_text) < 10:
            return "I am not able or allowed to get content from this URL."

        hostname = parsed_url.hostname
        current_date = datetime.now().strftime("%Y-%m-%d")
        metadata = {"url": url, "hostname": hostname, "date": current_date}
        metadata_delete = {"url": url}  # To delete previous content from the same URL
        self.utils.add_to_text_index(page_text, url, metadata, metadata_delete)
        
        if title is not None and len(title) > 0:
            summary = self.retrieve_from_archive(title + " - " + url)
            print(f"Summary length: {len(summary)}")
        else:
            summary = ""

        keywords_content = self.retrieve_from_archive(keywords)
        print(f"Keywords content length: {len(keywords_content)}")

        output = f"""The full content of the URL ({len(page_text)} characters) has been stored in the archive.
            Retrieve more information from the archive using keywords or browse links to get more information.
            A summary of the page:
            {between_xml_tag(summary, 'summary')}
            Additional information based on your keywords:
            {between_xml_tag(keywords_content, 'info')}"""
        
        print(f"Output length: {len(output)}")

        return output

    def retrieve_from_archive(self, query: str) -> str:
        """
        Retrieve content from the archive based on given query.

        Args:
            query (str): The keywords to search for.
            state (dict): The current state of the chat interface.

        Returns:
            str: XML-tagged output containing the search results as a JSON string.

        Note:
            This function uses the utils.get_text_catalog_search method to search the text index.
            It also keeps track of the archive ids in the state to avoid duplicates.
        """
        text_embedding = self.utils.get_embedding(input_text=query)

        query = {
            "size": self.config.MAX_ARCHIVE_RESULTS,
            "query": {"knn": {"embedding_vector": {"vector": text_embedding, "k": 5}}},
            "_source": ["date", "url", "hostname", "document"],
        }

        try:
            response = self.utils.get_text_catalog_search(query)
        except Exception as ex:
            error_message = f"Error: {ex}"
            print(error_message)
            return error_message            

        documents = ""
        for value in response["hits"]["hits"]:
            id = value["_id"]
            source = value["_source"]
            if id not in self.state["archive"]:
                documents += between_xml_tag(json.dumps(source), "document", {"id": id}) + "\n"
                self.state["archive"].add(id)

        print(f"Retrieved documents length: {len(documents)}")
        return between_xml_tag(documents, "documents")

    def get_tool_result_retrive_from_archive(self, tool_input: dict) -> str:
        """
        Retrieve content from the archive based on given keywords.

        This function searches the text index using the provided keywords and returns
        the matching documents.

        Args:
            tool_input (dict): A dictionary containing the 'keywords' to search for.

        Returns:
            str: XML-tagged output containing the search results as a JSON string.

        Note:
            This function uses the utils.get_text_catalog_search method to search the text index.
            It also keeps track of the archive ids in the state to avoid duplicates.
        """
        keywords = tool_input["keywords"]
        print(f"Keywords: {keywords}")

        return self.retrieve_from_archive(keywords)

    def get_tool_result_store_in_archive(self, tool_input: dict) -> str:
        """
        Store content in the archive.

        This function takes the provided content and stores it in the text index
        with the current date as metadata.

        Args:
            tool_input (dict): A dictionary containing the 'content' key with the text to be stored.
    
        Returns:
            str: A message indicating whether the content was successfully stored or an error occurred.

        Note:
            This function uses the utils.add_to_text_index method to store the content in the archive.
        """
        content = tool_input["content"]
        if len(content) == 0:
            return "You need to provide content to store in the archive."
        else:
            print(f"Content:\n---\n{content}\n---")

        current_date = datetime.now().strftime("%Y-%m-%d")
        metadata = {"date": current_date}
        id = uuid.uuid4()
        self.utils.add_to_text_index(content, id, metadata)

        return "The content has been stored in the archive."

    def render_sketchbook(self, sketchbook: list[str]) -> str:
        """
        Render a sketchbook as a single string, optionally using a new path for images.

        Args:
            sketchbook (list[str]): A list of strings, each representing a section in the sketchbook.

        Returns:
            str: A single string containing all sketchbook sections, properly formatted.
        """
        
        processed_sketchbook = [self.utils.process_image_placeholders_for_file(section) for section in sketchbook]

        rendered_sketchbook = "\n\n".join(processed_sketchbook)
        rendered_sketchbook = "\n" + re.sub(r'\n{3,}', '\n\n', rendered_sketchbook) + "\n"
        return rendered_sketchbook

    def get_tool_result_sketchbook(self, tool_input: dict) -> str:
        """
        Process a sketchbook command and update the sketchbook state accordingly.

        This function handles various sketchbook operations such as starting a new sketchbook,
        adding sections, reviewing sections, updating sections, and saving the sketchbook.

        Args:
            tool_input (dict): A dictionary containing the command and optional content.

        Returns:
            str: A message indicating the result of the operation.

        Commands:
            - start_new: Initializes a new empty sketchbook.
            - add_section: Adds a new section to the sketchbook.
            - start_review: Begins a review of the sketchbook from the first section.
            - next_section: Moves to the next section during review.
            - update_section: Updates the content of the current section.
            - save_sketchbook_file: Saves the sketchbook to a file.
            - info: Provides information about the sketchbook and current section.

        Note:
            This function uses the utils.render_sketchbook method to render the sketchbook.
            It also keeps track of the sketchbook sections in the state.
        """
        command = tool_input.get("command")
        content = tool_input.get("content", "")
        filename = tool_input.get("filename", "")
        print(f"Command: {command}")
        if len(content) > 0:
            print(f"Content:\n---\n{content}\n---")
        if len(filename) > 0:
            print(f"Filename: {filename}")

        num_sections = len(self.state["sketchbook"])

        match command:
            case "start_new":
                self.state["sketchbook"] = []
                self.state["sketchbook_current_section"] = 0
                return "This is a new sketchbook. There are no sections. Start by adding some content."
            case "add_section_at_the_end":
                if len(content) == 0:
                    return "You need to provide content to add a new section."
                try:
                    _ = self.utils.process_image_placeholders_for_file(content)
                except Exception as e:
                    error_message = f"Section not added. Error: {e}"
                    print(error_message)
                    return error_message
                self.state["sketchbook"].append(content)
                num_sections = len(self.state["sketchbook"])
                self.state["sketchbook_current_section"] = num_sections - 1
                return f"New section added at the end. You're now at section {self.state['sketchbook_current_section'] + 1} of {num_sections}. Add more sections, start a review, or save the sketchbook for the user."
            case "start_review":
                if num_sections == 0:
                    return "The sketchbook is empty. There are no sections to review or update. Start by adding some content."
                self.state["sketchbook_current_section"] = 0
                section_content = self.state["sketchbook"][0]
                section_content_between_xml_tag = between_xml_tag(section_content, "section")
                return f"You're starting your review at section 1 of {num_sections}. This is the content of the current section:\n\n{section_content_between_xml_tag}\n\nUpdate the content of this section, delete the section, or go to the next section. The review is completed when you reach the end."
            case "next_section":
                if self.state["sketchbook_current_section"] >= num_sections - 1:
                    return f"You're at the end. You're at section {self.state['sketchbook_current_section'] + 1} of {num_sections}. Start a review or save the sketchbook for the user."
                self.state["sketchbook_current_section"] += 1
                section_content = self.state["sketchbook"][self.state["sketchbook_current_section"]]
                section_content_between_xml_tag = between_xml_tag(section_content, "section", {"id": self.state["sketchbook_current_section"]})
                return f"Moving to the next section. You're now at section {self.state['sketchbook_current_section'] + 1} of {num_sections}. This is the content of the current section:\n\n{section_content_between_xml_tag}\n\nUpdate the content of this section, delete the section, or go to the next section. The review is completed when you reach the end."
            case "update_current_section":
                if num_sections == 0:
                    return "The sketchbook is empty. There are no sections. Start by adding some content."
                if len(content) == 0:
                    return "You need to provide content to update the current section."
                try:
                    _ = self.utils.process_image_placeholders_for_file(content)
                except Exception as e:
                    error_message = f"Section not updated. Error: {e}"
                    print(error_message)
                    return error_message
                self.state["sketchbook"][self.state["sketchbook_current_section"]] = content
                return f"The current section has been updated with the new content."
            case "delete_current_section":
                if num_sections == 0:
                    return "The sketchbook is empty. There are no sections to delete."
                self.state["sketchbook"].pop(self.state["sketchbook_current_section"])
                num_sections = len(self.state["sketchbook"])
                if num_sections == 0:
                    return "The section has been deleted. The sketchbook is now empty."
                if self.state["sketchbook_current_section"] >= num_sections - 1:
                    self.state["sketchbook_current_section"] -= 1
                section_content = self.state["sketchbook"][self.state["sketchbook_current_section"]]
                section_content_between_xml_tag = between_xml_tag(section_content, "section", {"id": self.state["sketchbook_current_section"]})
                return f"The section has been deleted. You're now at section {self.state['sketchbook_current_section'] + 1} of {num_sections}. This is the content of the current section:\n\n{section_content_between_xml_tag}\n\nUpdate the content of this section, delete the section, or go to the next section. The review is completed when you reach the end."
            case "share_sketchbook_as_a_file":
                if num_sections == 0:
                    return "The sketchbook is empty. There are no sections to save."
                if len(filename) == 0:
                    return "Provide a filename to save the sketchbook as a file."
                print("Saving the sketchbook...")
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                sketchbook_filename_without_extension = f"{filename}_{current_datetime}"
                sketchbook_full_absolute_path_without_extension = os.path.abspath(os.path.join(self.config.OUTPUT_PATH, sketchbook_filename_without_extension))
                try:
                    sketchbook_output = self.render_sketchbook(self.state["sketchbook"])
                except ImageNotFoundError as e:
                    return str(e)
                
                # First Markdown, then other formats
                response = ""
                input_format = "md"
                output_filename = f"{sketchbook_full_absolute_path_without_extension}.{input_format}"
                output_basename = os.path.basename(output_filename)
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write(sketchbook_output)
                    response += f"The sketchbook has been saved as {output_basename}.\n"
                except Exception as e:
                    error_message = f"Error: {e}"
                    print(error_message)
                    response += f"Error while saving the sketchbook as {input_format}: {error_message}\n"
                    return response

                # Now proceed with saving as DOCX
                output_format = "docx"
                output_filename = f"{sketchbook_full_absolute_path_without_extension}.{output_format}"
                output_basename = os.path.basename(output_filename)
                try:
                    pypandoc.convert_text(
                        sketchbook_output,
                        output_format,
                        format=input_format,
                        outputfile=output_filename,
                        extra_args=["--toc"],
                        cworkdir=self.config.OUTPUT_PATH,
                    )
                except Exception as e:
                    error_message = f"Error: {e}"
                    print(error_message)
                    response += f"Error while saving the sketchbook as {output_format}: {error_message}\n"
                    return response
                                                        
                return f"The sketchbook has been saved as {output_basename}.\nYou must now share the file with the user adding a line like this:\n[file: {output_basename}]"
            case _:
                return "Invalid command."

    def get_tool_result_checklist(self, tool_input: dict) -> str:
        """
        Process a checklist command and update the checklist state accordingly.

        This function handles various checklist operations such as starting a new checklist,
        adding items, showing items, and marking items as completed.

        Args:
            tool_input (dict): A dictionary containing the 'command' and optional 'content'.

        Returns:
            str: A message indicating the result of the operation.

        Commands:
            - start_new: Initializes a new empty checklist.
            - add_item_at_the_end: Adds a new item to the checklist.
            - show_items: Shows all items in the checklist.
            - mark_next_to_do_item_as_completed: Marks the next to-do item in the checklist as completed.

        Note:
            This function uses the utils.render_checklist method to render the checklist.
            It also keeps track of the checklist items in the state.
        """

        def render_checklist(render_between_xml_tag: bool = True) -> str:
            checklist = self.state["checklist"]
            output = ""
            for index, item in enumerate(checklist):
                item_state = "COMPLETED" if item['completed'] else "TO DO"
                output += f"{index + 1}. [{item_state}] {item['content']}\n"
            return between_xml_tag(output, "checklist") if render_between_xml_tag else output

        command = tool_input.get("command")
        content = tool_input.get("content", "")

        print(f"Command: {command}")
        if len(content) > 0:
            print(f"Content:\n---\n{content}\n---")

        num_items = len(self.state["checklist"])

        match command:
            case "start_new":
                self.state["checklist"] = []
                return "This is a new checklist. There are no items. Start by adding some content."
            case "add_item_as_next":
                if len(content) == 0:
                    return "You need to provide content to add a new item."
                item = {
                    "content": content,
                    "completed": False
                }
                self.state["checklist"].insert(0, item)
                return f"New item added as next. Add more items or mark the next to-do item as completed. These are the items in the current checklist:\n\n{render_checklist()}"
            case "add_item_at_the_end":
                if len(content) == 0:
                    return "You need to provide content to add a new item."
                item = {
                    "content": content,
                    "completed": False
                }
                self.state["checklist"].append(item)
                print(f"Checklist:\n{render_checklist(render_between_xml_tag=False)}")
                return f"New item added at the end. Add more items or mark the next to-do item as completed. These are the items in the current checklist:\n\n{render_checklist()}"
            case "show_items":
                if num_items == 0:
                    return "The checklist is empty. There are no items to show."
                print(f"Checklist:\n{render_checklist(render_between_xml_tag=False)}")
                return f"These are the items in the current checklist:\n\n{render_checklist()}"
            case "mark_next_to_do_item_as_completed":
                if num_items == 0:
                    return "The checklist is empty. There are no items to mark as completed."
                to_do_index = next((i for i, item in enumerate(self.state["checklist"]) if not item["completed"]), None)
                if to_do_index is None:
                    return "All items in the checklist are already completed."
                self.state["checklist"][to_do_index]["completed"] = True
                print(f"Checklist:\n{render_checklist(render_between_xml_tag=False)}")
                return f"Item number {to_do_index + 1} has been marked as completed. Add more items or mark the next to-do item as completed. These are the items in the current checklist:\n\n{render_checklist()}"
            case _:
                error_message = f"Invalid command: {command}"
                print(error_message)
                return error_message

    def get_tool_result_generate_image(self, tool_input: dict) -> str:
        """
        Generate an image based on a given prompt using Amazon Bedrock's image generation model.

        This function takes a text prompt, prepares the request body for the image generation API,
        invokes the model, and processes the response. If successful, it stores the generated image
        in the image catalog and adds it to the output state for display.

        Args:
            tool_input (dict): A dictionary containing the 'prompt' key with the text description for image generation.

        Returns:
            str: A message describing the generated image, including its ID and description.
                If an error occurs, it returns an error message instead.

        Note:
            This function uses the utils.generate_image method to generate the image.
            It also keeps track of the generated image in the state.
        """
        prompt = tool_input["prompt"]
        print(f"Prompt: {prompt}")

        try:
            response_body = self.utils.generate_image(prompt)
        except Exception as ex:
            error_message = f"Image generation error. Error is {ex}"
            print(error_message)
            return error_message

        finish_reason = response_body.get("error")

        if finish_reason is not None:
            error_message = f"Image generation error. Error is {finish_reason}"
            print(error_message)
            return error_message

        image_base64 = response_body.get("images")[0]
        image_format = "png"

        print(f"Image base64 size: {len(image_base64)}")
        image = self.utils.store_image(image_format, image_base64)

        return f"A new image with with 'image_id' {image['id']} and this description has been stored in the image catalog:\n\n{image['description']}\nDon't mention the 'image_id' in your response."

    def get_tool_result_search_image_catalog(self, tool_input: dict) -> str:
        """
        Search for images in the image catalog based on a text description.

        This function retrieves images from the catalog that match a given description,
        adds them to the output state for display, and prepares a summary of the results.

        Args:
            tool_input (dict): A dictionary containing:
                - description (str): The text description to search for.
                - max_results (int, optional): Maximum number of results to return.
                Defaults to MAX_IMAGE_SEARCH_RESULTS.

        Returns:
            str: A summary of the search results, including descriptions of found images.
                If no images are found, returns a message indicating so.

        Note:
            This function uses the utils.get_images_by_description method to find images.
            It also keeps track of the found images in the state.
        """
        description = tool_input.get("description")
        max_results = tool_input.get("max_results", self.config.MAX_IMAGE_SEARCH_RESULTS)
        print(f"Description: {description}")
        print(f"Max results: {max_results}")
        images = self.utils.get_images_by_description(description, max_results)
        if type(images) is not list:
            return images  # It's an error
        result = ""
        for image in images:
            result += f"Found image with 'image_id' {image['id']} and this description:\n\n{image['description']}\n\n"
        if len(result) == 0:
            return f"No images found."
        result = f"These are images similar to the description in descreasing order of similarity:\n{result}"
        return result

    def get_tool_result_similarity_image_catalog(self, tool_input: dict) -> str:
        """
        Search for similar images in the image catalog based on a reference image.

        This function retrieves images from the catalog that are similar to a given reference image,
        adds them to the output state for display, and prepares a summary of the results.

        Args:
            tool_input (dict): A dictionary containing:
                - image_id (str): The ID of the reference image to search for similar images.
                - max_results (int, optional): Maximum number of results to return.
                Defaults to MAX_IMAGE_SEARCH_RESULTS.

        Returns:
            str: A summary of the search results, including descriptions of found images.
                If no similar images are found, returns a message indicating so.
                If an error occurs, returns an error message.

        Note:
            This function uses the utils.get_images_by_similarity method to find similar images.
            It also keeps track of the found images in the state.
        """
        image_id = tool_input.get("image_id")
        max_results = tool_input.get("max_results", self.config.MAX_IMAGE_SEARCH_RESULTS)
        print(f"Image ID: {image_id}")
        print(f"Max results: {max_results}")
        similar_images = self.utils.get_images_by_similarity(image_id, max_results)
        if type(similar_images) is not list:
            return similar_images  # It's an error
        result = ""
        for image in similar_images:
            assert image_id != image["id"]
            result += f"Found image with 'image_id' {image['id']} and this description:\n\n{image['description']}\n\n"
        if len(result) == 0:
            return f"No similar images found."
        result = f"These are images similar to the reference image in descreasing order of similarity:\n{result}"
        return result

    def get_tool_result_random_images(self, tool_input: dict) -> str:
        """
        Retrieve random images from the image catalog and add them to the output state.

        This function fetches a specified number of random images from the image catalog,
        adds them to the output state for display, and prepares a summary of the results.

        Args:
            tool_input (dict): A dictionary containing:
                - num (int): The number of random images to retrieve.

        Returns:
            str: A summary of the random images retrieved, including descriptions of each image.
                If no images are returned or an error occurs, returns an appropriate message.

        Note:
            This function uses the utils.get_random_images method to find images.
        """
        num = tool_input.get("num")
        print(f"Num: {num}")
        random_images = self.utils.get_random_images(num)
        if type(random_images) is not list:
            return random_images  # It's an error
        result = ""
        for image in random_images:
            result += f"Found image with 'image_id' {image['id']} and this description:\n\n{image['description']}\n\n"
        if len(result) == 0:
            return f"No random images returned."
        result = f"These are random images from the image catalog:\n{result}"
        return result

    def get_tool_result_get_image_by_id(self, tool_input: dict) -> str:
        """
        Get info on an image by its 'image_id'. This tool can also be used to check that an image_id is valid and exists in the image catalog.

        Args:
            tool_input (dict): A dictionary containing the 'image_id' key with the image ID.

        Returns:
            str: A message describing the image, including its ID and description.
        """
        image_id = tool_input.get("image_id")
        print(f"Image ID: {image_id}")
        image = self.utils.get_image_by_id(image_id)
        if type(image) is not dict:
            return f"Error: Image not found."  # It's an error
        return f"Found image with 'image_id' {image['id']} and description:\n\n{image['description']}"

    def get_tool_result_image_catalog_count(self, _tool_input: dict) -> int | str:
        """
        Count the number of documents in the image catalog.

        This function queries the OpenSearch index to get the total count of images
        in the multimodal index.

        Args:
            _tool_input (dict): The tool input (unused in this function).

        Returns:
            int: The number of images in the catalog.
            str: An error message if an exception occurs during the count operation.

        Note:
            This function uses the utils.get_image_catalog_count_info method to get the count of images.
        """
        try:
            info = self.utils.get_image_catalog_count_info()
            print(f"Image catalog info: {info}")
            count = info["count"]
            return count
        except Exception as ex:
            error_message = f"Error: {ex}"
            print(error_message)
            return error_message

    def get_tool_result_download_image_into_catalog(self, tool_input: dict) -> str:
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

        Returns:
            str: A message describing the result of the operation, including the image ID
                and description if successful, or an error message if the operation fails.

        Raises:
            Various exceptions may be raised and caught within the function, resulting
            in error messages being returned instead of the function terminating.

        Note:
            This function uses the utils.get_image_base64 method to download the image.
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

        if type != 'image' or format not in self.config.IMAGE_FORMATS:
            error_message = f"Unsupported image format: {content_type}"
            print(error_message)
            return error_message

        # Get file extension from URL
        try:
            image_base64 = self.utils.get_image_base64(url,
                                            format=format,
                                            max_image_size=self.config.MAX_CHAT_IMAGE_SIZE,
                                            max_image_dimension=self.config.MAX_CHAT_IMAGE_DIMENSIONS)
        except Exception as ex:
            error_message = f"Error downloading image: {ex}"
            print(error_message)
            return error_message

        # Convert image bytes to base64
        image = self.utils.store_image(format, image_base64)

        print(f"Image stored: {image}")

        return f"Image downloaded and stored in the image catalog with 'image_id' {image['id']} and description:\n\n{image['description']}"

    def get_tool_result_personal_improvement(self, tool_input: dict) -> str:
        """
        Handle personal improvement commands and update the improvement state.

        This function processes commands related to personal improvements, including
        showing current improvements and updating them.

        Args:
            tool_input (dict): A dictionary containing command information:
                - command (str): The action to perform ('show_improvements' or 'update_improvements').
                - improvements (str, optional): New improvements to be stored, replacing the current ones.

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
                return f"These are the current improvements:\n{self.state['improvements']}"
            case 'update_improvements':
                self.state['improvements']= improvements
                return"Improvement updated."
            case _:
                return "Invalid command."

    def get_tool_result_arxiv(self, tool_input: dict) -> str:
        """
        Search for papers on arXiv.

        This function searches for papers on arXiv, downloads them, and adds them to the text index.
        It uses the arxiv library to search for papers and the process_pdf_document function to extract text from the papers.
        The function then adds the text to the text index.

        Args:
            tool_input (dict): A dictionary containing the search query.

        Returns:
            str: A message indicating the result of the operation.
        """
        query = tool_input.get("query")
        max_results = int(tool_input.get("max_results", self.config.MAX_ARXIV_RESULTS))
        print(f"Query: {query}")
        print(f"Max results: {max_results}")

        arxiv_client = arxiv.Client()
        search = arxiv.Search(
            query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        abstracts = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, result in enumerate(arxiv_client.results(search)):
                print(f"Downloading result {i}: {result.entry_id} - {result.title} ...")
                abstracts[result.entry_id] = result.title + "\n\n" + result.summary
                filename = f"{i}.pdf"
                try:
                    result.download_pdf(dirpath=temp_dir, filename=filename)
                except Exception as ex:
                    error_message = f"Error downloading PDF with filename '{filename}': {ex}"
                    print(error_message)
                    print(f"Skipping this result because it might have been withdrawn.")
                    continue

                full_path = os.path.join(temp_dir, filename)
                article_text = result.title + "\n\n" + self.utils.process_pdf_document(full_path)

                print(f"Output length: {len(article_text)}")
                current_date = datetime.now().strftime("%Y-%m-%d")
                metadata = {"arxiv": result.entry_id, "date": current_date}
                metadata_delete = {"arxiv": result.entry_id}
                self.utils.add_to_text_index(article_text, result.entry_id, metadata, metadata_delete)

        all_abstracts = ""
        for id, abstract in abstracts.items():
            all_abstracts += f"{between_xml_tag(abstract, 'abstract', {'id': id})}\n"

        all_abstracts = between_xml_tag(all_abstracts, 'abstracts')

        print(f"Abstracts length: {len(all_abstracts)}")

        return f"Based on your query, I found the following articles on arXiv:\n\n{all_abstracts}\n\nThe full content of the articles has been stored in the archive. Now you must retrieve the information you need from each article in the archive."

    def get_tool_result_save_text_file(self, tool_input: dict) -> str:
        """
        Save a text file in output.

        Args:
            tool_input (dict): A dictionary containing the filename and content.

        Returns:
            str: A message indicating the result of the operation.

        Raises:
            Exception: If the filename already exists.

        This function saves a text file with the given filename and content.
        """
        filename = tool_input.get("filename")
        content = tool_input.get("content")
        print(f"Filename: {filename}")
        print(f"Content: {content}")
        
        try:
            full_path = os.path.join(self.config.OUTPUT_PATH, filename)
            if os.path.exists(full_path):
                return f"File already exists: {filename}. Please choose a different filename."
            with open(full_path, 'w') as f:
                f.write(content)
        except Exception as ex:
            error_message = f"Error saving file: {ex}"
            print(error_message)
            return error_message

        return f"File saved: {filename}"

    def get_tool_result_check_if_file_exists(self, tool_input: dict) -> str:
        """
        Check if a file exists in the output folder.

        Args:
            tool_input (dict): A dictionary containing the filename.

        Returns:
            str: A message indicating the result of the operation.
        """
        filename = tool_input.get("filename")
        full_path = os.path.join(self.config.OUTPUT_PATH, filename)
        if os.path.exists(full_path):
            return f"File exists: {filename}"
        else:
            return f"File does not exist: {filename}"

    def get_tool_result_conversation(self, tool_input: dict) -> str:
        """
        Transform content into a conversation between three people where they describe the content by asking and answering questions.

        Args:
            tool_input (dict): A dictionary containing the conversation content.

        Returns:
            str: A message indicating the result of the operation.

        Raises:
            Exception: If the conversation content is not valid JSON.
        """
        num_participants = tool_input.get("num_participants")
        conversation = tool_input.get("conversation", "")
        filename = tool_input.get("filename", "")

        print(f"Number of participants: {num_participants}")
        print(f"Conversation: {conversation}")
        print(f"Filename: {filename}")

        if len(conversation) == 0:
            return "You need to provide some content in input."
        
        if len(filename) == 0:
            return "You need to provide a filename for the output audio file."

        try:
            conversation_lines = json.loads(conversation)
        except json.JSONDecodeError:
            raise ToolError("The output conversation is not a valid JSON")

        # conversation_lines = [ { "name": "John", "line": "Hello, how are you?" }, { "name": "Jane", "line": "I'm fine, thank you." } ]
        # Get unique names from the conversation
        unique_names = set(line["name"] for line in conversation_lines if "name" in line)

        # Create random panning for each voice
        pan_range = self.config.CONVERSATION_PAN_RANGE
        panning = {}
        positive_count = 0
        negative_count = 0
        for name in unique_names:
            if abs(positive_count - negative_count) <= 1:
                pan_value = random.uniform(-pan_range, pan_range)
            elif positive_count > negative_count:
                pan_value = random.uniform(-pan_range, 0)
            else:
                pan_value = random.uniform(0, pan_range)
            
            panning[name] = pan_value

            if pan_value >= 0:
                positive_count += 1
            else:
                negative_count += 1
        print(f"Panning: {panning}")

        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_without_extension = os.path.join(self.config.OUTPUT_PATH, f"{filename}_{current_datetime}")

        conversation_text = ""
        for full_line in conversation_lines:
            name = full_line.get("name")
            if name is None or name not in self.config.CONVERSATION_VOICES:
                raise ToolError(f"Invalid voice: {name}")
            line = full_line.get("line")
            if line is None or len(line) == 0:
                raise ToolError(f"Invalid line: {line}")
            conversation_text += f"{name}: {self.utils.remove_all_xml_tags(line)}\n"
        open(f"{filename_without_extension}.txt", "w").write(conversation_text)

        audio_segments = [AudioSegment.empty() for _ in range(len(conversation_lines))]

        def process_script_line(index, full_line):
            name = full_line.get("name")
            line = full_line.get("line")
            
            print(f"Processing line {index + 1}: {name}: {line}")

            try:
                audio_data = self.utils.synthesize_speech(line, name, use_ssml=True)
                full_audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data)).pan(panning[name])
                audio_segments[index] = full_audio_segment
            except Exception as ex:
                raise ToolError(f"Error synthesizing speech: {ex}")
            
            return index

        indexes = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [executor.submit(process_script_line, i, full_line) for i, full_line in enumerate(conversation_lines)]
            for future in concurrent.futures.as_completed(futures):
                index = future.result()
                indexes.add(index)

        if len(indexes) != len(conversation_lines):
            raise ToolError("Some lines were not processed")

        full_audio_segment = AudioSegment.empty()
        for segment in audio_segments:
            full_audio_segment += segment

        output_filename = f"{filename_without_extension}.mp3"
        full_audio_segment.export(output_filename, format="mp3")

        output_basename = os.path.basename(output_filename)

        return f"The output conversation has been saved as an audio file ({output_basename}, {full_audio_segment.duration_seconds} seconds).\nYou must now share the file with the user adding a line like this:\n[file: {output_basename}]"

    def check_tools_consistency(self) -> None:
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
        tools_set = set([ t['toolSpec']['name'] for t in self.tools_json])
        tool_functions_set = set(self.tool_functions.keys())

        if tools_set != tool_functions_set:
            raise Exception(f"Tools and tool functions are not consistent: {tools_set} != {tool_functions_set}")

    def get_tool_result(self, tool_use_block: dict) -> str:
        """
        Execute a tool and return its result.

        Args:
            tool_use_block (dict): A dictionary containing the tool use information,
                               including the tool name and input.

        Returns:
            str: The result of the tool execution.

        Raises:
            ToolError: If an invalid tool name is provided or if there's an error during tool execution.
        """
        tool_use_name = tool_use_block['name']

        print(f"Tool: {tool_use_name}")

        try:
            return self.tool_functions[tool_use_name](tool_use_block['input'])
        except KeyError:
            raise ToolError(f"Invalid function name: {tool_use_name}")
