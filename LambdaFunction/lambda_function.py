import base64
import json
import os
import re
import subprocess

TMP_DIR = "/tmp"

# To avoid "Matplotlib created a temporary cache directory..." warning
os.environ['MPLCONFIGDIR'] = os.path.join(TMP_DIR, f'matplotlib_{os.getpid()}')

def remove_tmp_contents():

    # Traverse the /tmp directory tree
    for root, dirs, files in os.walk(TMP_DIR, topdown=False):
        # Remove files
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        
        # Remove empty directories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")


def lambda_handler(event, context):
    
    # Before
    remove_tmp_contents()

    input_script = event.get('input_script', '')
    if len(input_script) == 0:
        return {
            'statusCode': 400,
            'body': 'Input script is required'
        }

    print(f"Script:\n{input_script}")
    
    result = subprocess.run(["python", "-c", input_script], capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Search for "Show image" lines and convert images to base64

    images = []

    output_lines = output.split('\n')
    for i, line in enumerate(output_lines):
        match = re.match(r"Show image '(/tmp/.+)'", line)
        if match:
            image_path = match.group(1)
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
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

    # After
    remove_tmp_contents()

    result = {
        'output': output,
        'images': images
    }

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
