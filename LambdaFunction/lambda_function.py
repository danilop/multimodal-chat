import base64
import json
import os
import subprocess
from typing import Dict, Any

TMP_DIR = "/tmp"

IMAGE_EXTENSIONS = ['png', 'jpeg', 'jpg', 'gif', 'webp']

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


def do_install_modules(modules: list[str], current_env: dict[str, str]) -> str:    
    """
    Install Python modules using pip.

    This function takes a list of module names and attempts to install them
    using pip. It handles exceptions for each module installation and prints
    any errors encountered.

    Args:
        modules (list[str]): A list of module names to install.
    """

    output = ''

    for module in modules:
        try:
            subprocess.run(["pip", "install", module], check=True)
        except Exception as e:
            print(f"Error installing {module}: {e}")

    if type(modules) is list and len(modules) > 0:
        current_env["PYTHONPATH"] = TMP_DIR
        try:
            _ = subprocess.run(f"pip install -U pip setuptools wheel -t {TMP_DIR} --no-cache-dir".split(), capture_output=True, text=True, check=True)
            for module in modules:
                _ = subprocess.run(f"pip install {module} -t {TMP_DIR} --no-cache-dir".split(), capture_output=True, text=True, check=True)
        except Exception as e:
            error_message = f"Error installing {module}: {e}"
            print(error_message)
            output += error_message

    return output


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

    output = ""
    current_env = os.environ.copy()

    # No need to go further if there is no script to run
    input_script = event.get('input_script', '')
    if len(input_script) == 0:
        return {
            'statusCode': 400,
            'body': 'Input script is required'
        }

    install_modules = event.get('install_modules', [])
    output += do_install_modules(install_modules, current_env)

    print(f"Script:\n{input_script}")
    
    result = subprocess.run(["python", "-c", input_script], env=current_env, capture_output=True, text=True)
    output += result.stdout + result.stderr

    # Search for images and convert them to base64
    images = []

    for file in os.listdir(TMP_DIR):
        file_path: str = os.path.join(TMP_DIR, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(f".{ext}") for ext in IMAGE_EXTENSIONS):
            try:
                # Read file content
                with open(file_path, "rb") as f:
                    file_content: bytes = f.read()
                    images.append({
                        "path": file_path,
                        "base64": base64.b64encode(file_content).decode('utf-8')
                    })
                output += f"File {file_path} loaded.\n"
            except Exception as e:
                output += f"Error loading {file_path}: {e}"

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
