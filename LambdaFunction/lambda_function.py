import base64
import json
import os
import re
import subprocess

from typing import Dict, Any, List

TMP_DIR = "/tmp"

# To avoid "Matplotlib created a temporary cache directory..." warning
os.environ['MPLCONFIGDIR'] = os.path.join(TMP_DIR, f'matplotlib_{os.getpid()}')


def remove_tmp_contents() -> None:
    """
    Remove all contents (files and directories) from the temporary directory.

    This function traverses the /tmp directory tree and removes all files and empty
    directories. It handles exceptions for each removal attempt and prints any
    errors encountered.
    """
    # Traverse the /tmp directory tree
    for root, dirs, files in os.walk(TMP_DIR, topdown=False):
        # Remove files
        for file in files:
            file_path: str = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        
        # Remove empty directories
        for dir in dirs:
            dir_path: str = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function handler that executes a Python script and processes its output.

    This function takes an input Python script, executes it, captures the output,
    and processes any generated images. It also handles temporary file management.

    Args:
        event (Dict[str, Any]): The event dict containing the Lambda function input.
        context (Any): The context object provided by AWS Lambda.

    Returns:
        Dict[str, Any]: A dictionary containing the execution results, including:
            - statusCode (int): HTTP status code (200 for success, 400 for bad request)
            - body (str): Error message in case of bad request
            - output (str): The combined stdout and stderr output from the script execution
            - images (List[Dict[str, str]]): List of dictionaries containing image data
    """
    # Before running the script
    remove_tmp_contents()

    input_script: str = event.get('input_script', '')
    if len(input_script) == 0:
        return {
            'statusCode': 400,
            'body': 'Input script is required'
        }

    print(f"Script:\n{input_script}")
    
    result: subprocess.CompletedProcess = subprocess.run(["python", "-c", input_script], capture_output=True, text=True)
    output: str = result.stdout + result.stderr

    # Search for "Show image" lines and convert images to base64
    images: List[Dict[str, str]] = []

    output_lines: List[str] = output.split('\n')
    for i, line in enumerate(output_lines):
        match: Optional[re.Match] = re.match(r"Show image '(/tmp/.+)'", line)
        if match:
            image_path: str = match.group(1)
            try:
                with open(image_path, "rb") as image_file:
                    image_data: bytes = image_file.read()
                    images.append({
                        "path": image_path,
                        "base64": base64.b64encode(image_data).decode('utf-8')
                    })
            except Exception as e:
                output_lines[i] = f"Error loading image {image_path}: {str(e)}"

    output = '\n'.join(output_lines)

    print(f"Output: {output}")
    print(f"Len: {len(output)}")
    print(f"Images: {len(images)}")

    # After running the script
    remove_tmp_contents()

    result: Dict[str, Any] = {
        'output': output,
        'images': images
    }

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
