# Yet Another Intelligent Assistant (YAIA) 

A multimodal chat interface with access to many tools.

## Description

YAIA is a sophisticated multimodal chat interface powered by advanced AI models and equipped with a variety of tools. It can:

- Search and browse the web in real-time
- Query Wikipedia for information
- Perform news and map searches
- Safely execute Python code that can produce text and images such as charts and diagrams
- Compose long-form articles mixing text and images
- Generate, search, and compare images
- Analyze documents and images
- Search and download arXiv papers
- Generate and save conversations as text and audio files
- Save files to the output directory
- Track personal improvements
- Manage checklists for task tracking

## Architecture

These are the main components:

- [Gradio](https://www.gradio.app) 5 for the web interface
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) to handle conversation and tool use
- [Anthropic Claude 3.5 Sonnet](https://aws.amazon.com/bedrock/claude/) as main model
- [Amazon Titan Text and Multimodal Embeddings](https://aws.amazon.com/bedrock/titan) models
- [Amazon Titan Image Generator](https://aws.amazon.com/bedrock/titan)
- [OpenSearch](https://opensearch.org) for text and multimodal indexes
- [Amazon Polly](https://aws.amazon.com/polly/) for voices
- [AWS Lambda](https://aws.amazon.com/lambda) for the code interpreter

## Examples

Here are examples of how to use various tools:

1. **Web Search**:
   "Search the web for recent advancements in quantum computing."

2. **Wikipedia**:
   "Find Wikipedia articles about the history of artificial intelligence."

3. **Python Scripting**:
   "Create a Python script to generate a bar chart of global CO2 emissions by country."

4. **Sketchbook**:
   "Start a new sketchbook and write an introduction about how to compute Pi with numerical methods."

5. **Image Generation**:
   "Generate an image of a futuristic city with flying cars and tall skyscrapers."

6. **Image Search**:
   "Search the image catalog for pictures of endangered species."

7. **arXiv Integration**:
   "Search for recent research papers on deep learning in natural language processing."

8. **Conversation Generation**:
   "Create a conversation between three experts discussing how to set up multimodal RAG."

9. **File Management**:
   "Save a summary of our discussion about climate change to a file named 'climate_change_summary.txt'."

10. **Personal Improvement**:
    "Here's a suggestion to improve: to improve answers, search for official sources."

11. **Checklist**:
    "Start a new checklist to follow a list of tasks one by one."

## Key Features and Tools

1. **Web Interaction**:
   - DuckDuckGo Text Search: Performs web searches
   - DuckDuckGo News Search: Searches for recent news articles
   - DuckDuckGo Maps Search: Searches for locations and businesses
   - DuckDuckGo Images Search: Searches for publicly available images
   - Web Browser: Browses websites and retrieves their content

2. **Wikipedia Tools**:
   - Wikipedia Search: Finds relevant Wikipedia pages
   - Wikipedia Geodata Search: Locates Wikipedia articles by geographic location
   - Wikipedia Page Retriever: Fetches full Wikipedia page content

3. **Python Scripting**:
   - Runs Python scripts for computations, testing, and output generation, including text and images
   - Python modules can be added to the Python interpreter
   - Python code is run in a secure environment provided by AWS Lambda

4. **Content Management**:
   - Personal Archive: Stores and retrieves text, Markdown, or HTML content, using a semantic database
   - Sketchbook: Manages a multi-page sketchbook for writing and reviewing long-form content. Supports multiple output formats:
     - Markdown (.md): For easy reading and editing
     - Word Document (.docx): For document editing

5. **Image Handling**:
   - Image Generation: Creates images based on text prompts
   - Image Catalog Search: Searches images by description
   - Image Similarity Search: Finds similar images based on a reference image
   - Random Images: Retrieves random images from the catalog
   - Get Image by ID: Retrieves a specific image from the catalog using its ID
   - Image Catalog Count: Returns the total number of images in the catalog
   - Download Image: Adds images from URLs to the catalog

6. **arXiv Integration**:
   - Search and download arXiv papers
   - Store paper content in the archive for easy retrieval

7. **Conversation Generation**:
   - Transform content into a conversation between two to four people
   - Generate audio files for the conversation using text-to-speech

8. **File Management**:
   - Save File: Allows saving text content to a file with a specified name in the output directory

9. **Personal Improvement**:
   - Track suggestions and mistakes for future enhancements

10. **Checklist**:
    - Manage task lists with the ability to add items, mark them as completed, and review progress

For a comprehensive list of available tools and their usage, refer to `./Config/tools.json`.

## Requirements

1. A container tool: Docker or Finch (to install Finch, follow the [instructions here](https://runfinch.com/))
2. Python 3.12 or newer
3. AWS account with appropriate permissions to access Amazon Bedrock, AWS Lambda, and Amazon ECR

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/danilop/multimodal-chat
   cd multimodal-chat
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the AWS Lambda function for code execution:
   ```
   cd LambdaFunction
   ./deploy_lambda_function.sh
   cd ..
   ```

5. To use Selenium for web browsing, install ChromeDriver. Using Homebrew:
   ```
   brew install --cask chromedriver
   ```

6. To output audio, install `ffmpeg`. Using Homebrew:
   ```
   brew install ffmpeg
   ```

## Setting up OpenSearch

You can either use a local OpenSearch instance or connect to a remote server. For local setup:

1. Navigate to the OpenSearch directory:
   ```
   cd OpenSearch/
   ```

2. Set the admin password (first-time setup), this step will create the `.env` file and the `opensearch_env.sh` files:
   ```
   ./set_password.sh
   ```

3. Start OpenSearch locally (it needs access to the `.env` file):
   ```
   ./opensearch_start.sh
   ```

4. Ensure OpenSearch (2 nodes + dashboard) starts correctly by checking the output

5. To update OpenSearch, download the new container images using this script:
   ```
   ./opensearch_update.sh
   ```

For remote server setup, update the client creation code in the main script.

To change password, you need to delete the container uisng `finch` or `docker` and then set a new password.

## Usage

Default models for text, images, and embeddings are in the `Config/config.ini` file. The models to use are specified using [Amazon Bedrock model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) or [cross-region inference profile IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference-support.html). You need permissions and access to these models as described in [Access foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html).

This section assumes OpenSearch is running locally in another terminal window as described before.

1. Load the OpenSearch admin password into the environment:
   ```
   source OpenSearch/opensearch_env.sh
   ```

2. Run the application:
   ```
   python multimodal_chat.py
   ```

3. To reset the text and multimodal indexes (note: this doesn't delete images in `./Images/`):
   ```
   python multimodal_chat.py --reset-index
   ```

4. Open a web browser and navigate to http://127.0.0.1:7860/ to start chatting.

## Demo videos

Here are a few examples of what you can do this application.

### Browse the internet and use the semantic archive

In this demo:
- Browse websites using Selenium and specific tools for DuckDuckGo (search, news, geosearch) and Wikipedia
- Use the semantic text archive tool to archive documents and retrieve by keywords

[![Multimodal Chat Demo 1 – Browse the internet and use the semantic archive](https://img.youtube.com/vi/HNBwQ3PEgWU/0.jpg)](https://www.youtube.com/watch?v=HNBwQ3PEgWU)

### Import and search Images

In this demo:
- Using a multimodal index and the local file system to manage an image catalog
- Store images with a generated description
- Retrieve images by text description (semantic search)
- Retrieve images by similarity to another image
- Retrieve random images

[![Multimodal Chat Demo 2 – Import and search images](https://img.youtube.com/vi/ytQbvXjjQF0/0.jpg)](https://www.youtube.com/watch?v=ytQbvXjjQF0)

### Generate and search images

In this demo:
- Generate images from a textual description
- The text-to-image prompt is generated from chat instructions
- This approach allows to use the overall conversation to improve the prompt

[![Multimodal Chat Demo 3 – Generate and search images](https://img.youtube.com/vi/QfTgaTB1TWE/0.jpg)](https://www.youtube.com/watch?v=QfTgaTB1TWE)

### Python code interpreter

In this demo:
- Running AI generated code to solve problems
- Running for security in an AWS Lambda function with basic permissions
- Deployed via a container image to easily add Python modules
- Python only but easily extensible

[![Multimodal Chat Demo 4 – Python code interpreter](https://img.youtube.com/vi/WcRMM3ulbTc/0.jpg)](https://www.youtube.com/watch?v=WcRMM3ulbTc)

### Writing on a "sketchbook"

In this demo:
- A tool to help write long forms of text such as articles and blog posts)
- Providing sequential access to text split in pages
- To mitigate the "asymmetry" between a model input and output sizes

[![Multimodal Chat Demo 5 – Writing on a "sketchbook"](https://img.youtube.com/vi/BZHdufdMlfI/0.jpg)](https://www.youtube.com/watch?v=BZHdufdMlfI)

### Sketchbook with a Python code review

In this demo:
- Best results use more than one tools together
- Start with a sketchbook to write a long article
- The article contains code snippets
- A review runs and tests all code snippets and updates each page fixing the code (if needed) and adding actual results

[![Multimodal Chat Demo 6 – Sketchbook with a Python code review](https://img.youtube.com/vi/6nGRFDsk-C4/0.jpg)](https://www.youtube.com/watch?v=6nGRFDsk-C4)

## Troubleshooting

- If you encounter issues with OpenSearch, check the connection settings and ensure the service is running
- For AWS Lambda function errors, verify your AWS credentials and permissions
- If image processing fails, ensure you have the necessary libraries installed and check file permissions

## Contributing

Contributions to YAIA are welcome! Please refer to the contributing guidelines for more information on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Usage Tips

- Combine multiple tools for complex tasks. For example, use the web search to find information, then use the sketchbook to write a summary, and finally generate a conversation about the topic.
- When working with images, you can generate new images, search for existing ones, or download images from the web to add to your catalog.
- Use the arXiv integration to stay up-to-date with the latest research in your field of interest.
- The conversation generation tool is great for creating engaging content or preparing for presentations.
- Regularly check and update your personal improvements to track your progress and areas for growth.

For more detailed information on specific components or advanced usage, please refer to the inline documentation in the source code.
