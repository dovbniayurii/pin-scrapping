import json
import pinecone
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import unicodedata

# Initialize Pinecone
pinecone.init(api_key="pcsk_6pkmwU_5WC2wWq5j4KGq8S1QgLGnGkByZroAcSHRKLDh2YnjgfssPqomenV6ZkAPv7SaxM")
index_name = "pin-collection-image-prod"
print("Pinecone initialized ----------- ", pinecone.list_indexes())

# Create a Pinecone index if it doesn't already exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=512)  # CLIP embeddings are 512-dimensional
index = pinecone.Index(index_name)

print("Pinecone index initialized ----------- ")

# Initialize CLIP model for embedding generation
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to sanitize the ID (ensure ASCII-compatible)
def sanitize_id(text):
    """Convert non-ASCII characters in a string to ASCII equivalents or remove them."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Function to preprocess and generate embeddings
def generate_embedding(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", size=224)
            outputs = model.get_image_features(**inputs)
            return outputs[0].detach().numpy().tolist()  # Convert to list for Pinecone
        else:
            print(f"Failed to fetch image: {image_url}")
            return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Load and index data from JSONL file
def index_pins(jsonl_file, error_file):
    with open(jsonl_file, "r") as file, open(error_file, "w") as error_log:
        for line in file:
            try:
                # Parse JSONL entry
                data = json.loads(line)
                image_url = data.get("image_url")

                # Sanitize the ID to ensure it is ASCII-compatible
                sanitized_id = sanitize_id(data["name"])

                # Validate image URL
                if not image_url:
                    error_message = f"Missing image URL. Skipping entry: {data}\n"
                    print(error_message)
                    error_log.write(error_message)
                    continue

                # Generate embedding
                embedding = generate_embedding(image_url)
                if embedding is None:
                    error_message = f"Skipping entry due to failed embedding for image: {image_url}\n"
                    print(error_message)
                    error_log.write(error_message)
                    continue

                # Ensure metadata values are valid (string, integer, float)
                metadata = {key: value for key, value in data.items() if isinstance(value, (str, int, float))}

                # Index into Pinecone
                index.upsert([(sanitized_id, embedding, metadata)])
                print(f"Indexed: {sanitized_id}")

            except Exception as e:
                error_message = f"Error processing entry: {line.strip()} | Exception: {str(e)}\n"
                print(error_message)
                error_log.write(error_message)

# Path to your JSONL file
jsonl_file_path = "./jsonl_output/pins_details.jsonl"
error_file_path = "./error.txt"

# Index pins
index_pins(jsonl_file_path, error_file_path)

print("Indexing completed successfully!")
