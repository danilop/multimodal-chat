import os
import subprocess

TMP_DIR = "/tmp"

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

    print(f"Output: {output}")
    print(f"Len: {len(output)}")

    # After
    remove_tmp_contents()

    return output
