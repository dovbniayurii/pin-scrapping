import os
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# Initialize Pinecone
api_key = "pcsk_6pkmwU_5WC2wWq5j4KGq8S1QgLGnGkByZroAcSHRKLDh2YnjgfssPqomenV6ZkAPv7SaxM"
pc = Pinecone(api_key=api_key)

index_name = "pin-collection-image-prod"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust this based on your use case
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Adjust region as needed
        )
    )

index = pc.Index(index_name)

# Initialize CLIP model for embedding generation
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate embedding for the test image
def generate_embedding(image_path_or_url):
    try:
        # Load image from URL or local file
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        # Generate embedding
        inputs = processor(images=image, return_tensors="pt", size=224)
        outputs = model.get_image_features(**inputs)
        return outputs[0].detach().numpy().tolist()  # Convert to list for querying
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Test image input (replace with your local image path or URL)
test_image_path = "1.png"  # Local image path

# Generate embedding for the test image
test_embedding = generate_embedding(test_image_path)

if test_embedding:
    # Perform similarity search in Pinecone
    search_results = index.query(vector=test_embedding, top_k=5, include_metadata=True)

    # Display the results
    print("Search Results:")
    for match in search_results["matches"]:
        print(f"ID: {match['id']}, Score: {match['score']}")
        print(f"Metadata: {match['metadata']}")
else:
    print("Failed to generate embedding for the test image.")