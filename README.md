# Yet Another Chatbot – A Multimodal Chat Interface

## Description

Yet Another Chatbot is a sophisticated multimodal chat interface powered by advanced AI models and equipped with a variety of tools. This chatbot can:

- Search and browse the web in real-time
- Query Wikipedia for information
- Perform news and map searches
- Safely execute Python code
- Compose long-form articles
- Generate, search, and compare images

## Key Features and Tools

1. **Web Interaction**:
   - DuckDuckGo Search: Performs web, news, and map searches
   - Web Browser: Browses websites and retrieves their content

2. **Wikipedia Tools**:
   - Search: Finds relevant Wikipedia pages
   - Geodata Search: Locates Wikipedia articles by geographic location
   - Page Retriever: Fetches full Wikipedia page content

3. **Python Scripting**: Runs Python scripts for computations, testing, and output generation

4. **Content Management**:
   - Personal Archive: Stores and retrieves text, Markdown, or HTML content
   - Notebook: Manages a multi-page notebook for writing and reviewing long-form content

5. **Image Handling**:
   - Image Generation: Creates images based on text prompts
   - Image Catalog: Searches, compares, and manages images
   - Image Downloader: Adds images from URLs to the catalog

6. **Self-Improvement**:
   - Personal Improvement Tracker: Tracks suggestions and mistakes for future enhancements

For a comprehensive list of available tools and their usage, refer to `./Config/tools.json`.

## Requirements

1. A container tool: Docker or Finch (To install Finch, follow the [instructions here](https://runfinch.com/))
2. Python 3.12 or newer
3. AWS account with appropriate permissions

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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

## Setting up OpenSearch

You can either use a local OpenSearch instance or connect to a remote server. For local setup:

1. Navigate to the OpenSearch directory:
   ```
   cd OpenSearch/
   ```

2. Set the admin password (first-time setup):
   ```
   ./set_password.sh
   ```

3. Start OpenSearch locally (it needs access to the `.env` file):
   ```
   ./opensearch_start.sh
   ```

4. Ensure OpenSearch (2 nodes + dashboard) starts correctly by checking the output

For remote server setup, update the client creation code in the main script.

## Usage

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

4. Open a web browser and navigate to http://127.0.0.1:7860/ to start chatting

## Examples

The `Examples` section in the chat interface provides sample queries to help you get started. These demonstrate various capabilities of the chatbot, including web searches, code execution, and image-related tasks.

## Troubleshooting

- If you encounter issues with OpenSearch, check the connection settings and ensure the service is running
- For AWS Lambda function errors, verify your AWS credentials and permissions
- If image processing fails, ensure you have the necessary libraries installed and check file permissions

## Contributing

Contributions to Yet Another Chatbot are welcome! Please refer to the contributing guidelines for more information on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For more detailed information on specific components or advanced usage, please refer to the inline documentation in the source code.
