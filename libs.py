import json
from urllib.parse import urlparse

import html2text


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


def between_xml_tag(text: str, element: str, attributes: dict|None = None) -> str:
    """
    Wrap the given text with specified XML tags.

    Args:
        text (str): The text to be wrapped.
        element (str): The XML element to use for wrapping without the angle brackets.
        attributes (dict, optional): Additional attributes to add to the element. Defaults to None.

    Returns:
        str: The text wrapped in the specified XML tags.

    Example:
        >>> with_xml_tag("Hello, World!", "greeting", {"class": "important"})
        '<greeting class="important">Hello, World!</greeting>'
    """
    if attributes is not None:
        attributes_str = ' '.join([f'{key}="{value}"' for key, value in attributes.items()])
        return f"<{element} {attributes_str}>{text}</{element}>"
    else:
        return f"<{element}>{text}</{element}>"


def get_base_url(url: str) -> str:
    """
    Extract the base URL from a given URL.

    This method takes a full URL and returns its base URL, which consists
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

    This method takes HTML text and a base URL, then converts the HTML to Markdown
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

