import requests
import os

# API endpoint
url = "http://localhost:8000/upload/"

# Path to the file you want to upload
# Assumes the script is in AGENTIC_MIRAI and test_docs is a subdirectory
#file_path = os.path.join("test_docs", "sample.txt")
file_path = os.path.join("/Users/ashish/Documents/Work/Main/BitBucket/agentic_mirai/test_docs/sample.txt")
# Or provide an absolute path: file_path = "/path/to/your/test_docs/sample.txt"


# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Data for other form fields (optional)
    payload_data = {
        'chunk_size': '256',  # Send as strings, FastAPI will convert
        'chunk_overlap': '30'
    }
    # If you don't want to send chunk_size/overlap, use:
    # payload_data = {}

    # Files to send (key is 'file', matching the FastAPI parameter)
    # The tuple is (filename, file_object, content_type)
    files_to_upload = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'text/plain')
    }
    # If you don't want to send chunk_size/overlap, use:
    # response = requests.post(url, files=files_to_upload)

    try:
        # Make the POST request
        response = requests.post(url, files=files_to_upload, data=payload_data)

        # Check the response
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        print("Upload successful!")
        print("Response JSON:", response.json())

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print("Response Content:", response.text)
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    finally:
        # Important: Close the file object
        if 'file' in files_to_upload and files_to_upload['file'][1]:
            files_to_upload['file'][1].close()