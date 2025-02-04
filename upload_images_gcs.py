import json
import requests
import os
from google.cloud import storage
from tqdm import tqdm

# Configuration
GCS_BUCKET_NAME = "pin_collection"
LOCAL_JSONL_FILE = "pins_details.jsonl"
UPDATED_JSONL_FILE = "updated_data.jsonl"
ERROR_LOG_FILE = "error_log.txt"
GCS_FOLDER = "images/"  # Folder inside GCS (optional)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def upload_to_gcs(image_url, filename, line_number):
    """Downloads image and uploads it to GCS, returning the public URL."""
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            error_msg = f"[Line {line_number}] Failed to download {image_url}"
            print(error_msg)
            log_error(error_msg)
            return None

        gcs_path = f"{GCS_FOLDER}{filename}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(response.content, content_type="image/jpeg")

        return blob.public_url
    except Exception as e:
        error_msg = f"[Line {line_number}] Error uploading {image_url}: {e}"
        print(error_msg)
        log_error(error_msg)
        return None

def log_error(message):
    """Logs errors to a file."""
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as error_log:
        error_log.write(message + "\n")

# Process JSONL file (force UTF-8 encoding)
with open(LOCAL_JSONL_FILE, "r", encoding="utf-8") as infile, open(UPDATED_JSONL_FILE, "w", encoding="utf-8") as outfile:
    for line_number, line in enumerate(tqdm(infile, desc="Processing JSONL"), start=1):
        try:
            data = json.loads(line.strip())
            image_url = data.get("image_url")
            if image_url:
                filename = os.path.basename(image_url)  # Extract filename
                new_url = upload_to_gcs(image_url, filename, line_number)
                if new_url:
                    data["image_url"] = new_url  # Update URL
            
            outfile.write(json.dumps(data) + "\n")
        except Exception as e:
            error_msg = f"[Line {line_number}] Error processing line: {e}"
            print(error_msg)
            log_error(error_msg)

print(f"Updated JSONL saved as {UPDATED_JSONL_FILE}")
print(f"Errors logged in {ERROR_LOG_FILE} if any.")
