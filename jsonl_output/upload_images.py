from google.cloud import storage
import requests
from tqdm import tqdm
import os

# Configurations
BUCKET_NAME = "your-bucket-name"
IMAGE_FOLDER = "images/"  # GCS folder

# Initialize GCS client
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Load image URLs from JSONL file
input_file = "pins_details.jsonl"

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

for line in tqdm(lines, desc="Uploading images"):
    obj = json.loads(line.strip())
    image_url = obj.get("image_url")
    if not image_url:
        continue  # Skip if no image

    # Download image
    image_name = os.path.basename(image_url)
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        blob = bucket.blob(f"{IMAGE_FOLDER}{image_name}")
        blob.upload_from_string(response.content, content_type="image/jpeg")
        blob.make_public()  # Make image public

        print(f"Uploaded: {blob.public_url}")  # Get public URL
