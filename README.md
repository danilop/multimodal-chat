# Yet Another Chatbot – A Multimodal Chat Interface

## Description

A multimodal chat interface with many tools.

It can search and browse the web, search Wikipedia, including news and maps, safely run Python code, write long articles, and generate, search, and compare images.

See `./Config/tools.json` for an overview of the available tools, and how they can be used by the model.

## Requirements

1. A container tool (either docker or finch)

2. Python (3.12 or newer tested)

## Installation

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up AWS Lambda function to run code for the model:
   ```
   cd LambdaFunction
   ./deploy_lambda_function.sh
   cd ..
   ```

## Start OpenSearch locally

These instructions start OpenSearch locally. You can use a remote server by updating the code creating the client. In that cae, you don't need these steps.

1. Enter in the `OpenSearch` directory:
   ```
   cd OpenSearch/
   ```

2. The first time, set up the admin password:
   ```
   ./set_password.sh
   ```

3. Start OpenSerach locally:
   ```
   ./opensearch_start.sh
   ```

4. Check from the output that OpenSearch (2 nodes + dashboard) starts correctly

## Usage

1. Load OpenSearch admin password in the environment:
   ```
   source OpenSearch/opensearch_env.sh        
   ```

1. Run the application:
   ```
   python multimodal_chat.py
   ```

2. To reset the text and multimodal indexes (this doesn't delete images under `./Images/`):
   ```
   python multimodal_chat.py --reset-index
   ```

3. Connect to http://127.0.0.1:7860/ and enjoy chatting! See the `Examples` section for soem ideas on what to ask.