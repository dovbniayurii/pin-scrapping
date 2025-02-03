import json
import requests
import os
from google.cloud import storage
from tqdm import tqdm

# Configuration
GCS_BUCKET_NAME = "pin_collection"
LOCAL_JSONL_FILE = "pins_details.jsonl"
UPDATED_JSONL_FILE = "updated_data.jsonl"
GCS_FOLDER = "images/"  # Folder inside GCS (optional)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def upload_to_gcs(image_url, filename):
    """Downloads image and uploads it to GCS, returning the public URL."""
    try:
        # Download image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download {image_url}")
            return None

        # Define GCS path
        gcs_path = f"{GCS_FOLDER}{filename}"
        blob = bucket.blob(gcs_path)

        # Upload image
        blob.upload_from_string(response.content, content_type="image/jpeg")
        blob.make_public()

        return blob.public_url
    except Exception as e:
        print(f"Error uploading {image_url}: {e}")
        return None

# Process JSONL file
with open(LOCAL_JSONL_FILE, "r") as infile, open(UPDATED_JSONL_FILE, "w") as outfile:
    for line in tqdm(infile, desc="Processing JSONL"):
        data = json.loads(line.strip())

        # Extract image URL
        image_url = data.get("image_url")
        if image_url:
            filename = os.path.basename(image_url)  # Extract filename
            new_url = upload_to_gcs(image_url, filename)

            if new_url:
                data["image_url"] = new_url  # Update URL

        outfile.write(json.dumps(data) + "\n")

print(f"Updated JSONL saved as {UPDATED_JSONL_FILE}")
