import os
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import Union, Optional, Dict, List
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_search.log'),
        logging.StreamHandler()
    ]
)

class ImageSearchTester:
    def __init__(self, api_key: str, index_name: str = "pin-collection-image-prod"):
        self.api_key = api_key
        self.index_name = index_name
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize Pinecone and CLIP model"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check or create index
            if self.index_name not in self.pc.list_indexes().names():
                logging.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=512,  # CLIP-base produces 512-dim embeddings
                    metric='cosine',  # Better for similarity search than Euclidean
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
            
            self.index = self.pc.Index(self.index_name)
            
            # Initialize CLIP model with error handling
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {self.device}")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            logging.info("Components initialized successfully")
            
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def load_image(self, image_source: Union[str, bytes]) -> Optional[Image.Image]:
        """Load image from URL, file path, or bytes"""
        try:
            if isinstance(image_source, bytes):
                return Image.open(BytesIO(image_source)).convert("RGB")
            elif image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif os.path.exists(image_source):
                return Image.open(image_source).convert("RGB")
            else:
                logging.error("Invalid image source provided")
                return None
        except Exception as e:
            logging.error(f"Failed to load image: {str(e)}")
            return None

    def generate_embedding(self, image: Image.Image) -> Optional[List[float]]:
        """Generate embedding with augmentation and normalization"""
        try:
            # Apply preprocessing
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            
            # Normalize the embedding
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
            
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            return None

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """Enhanced similarity search with filters"""
        try:
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            if filters:
                query_params["filter"] = filters
            
            results = self.index.query(**query_params)
            
            # Post-process results
            for match in results['matches']:
                match['score'] = float(match['score'])  # Convert numpy float to native float
                
            return results
            
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return {"matches": []}

    def visualize_results(self, results: Dict, query_image: Image.Image = None):
        """Display results with optional query image visualization"""
        from matplotlib import pyplot as plt
        
        if query_image:
            plt.figure(figsize=(5, 5))
            plt.imshow(query_image)
            plt.title("Query Image")
            plt.axis('off')
            plt.show()
        
        print("\nTop Matches:")
        for idx, match in enumerate(results['matches'], 1):
            print(f"\n#{idx}:")
            print(f"ID: {match['id']}")
            print(f"Similarity Score: {match['score']:.4f}")
            
            if 'image_url' in match['metadata']:
                print(f"Image URL: {match['metadata']['image_url']}")
            
            # Display other relevant metadata
            for k, v in match['metadata'].items():
                if k not in ['image_url', 'embedding'] and len(str(v)) < 100:
                    print(f"{k}: {v}")

    def test_search(
        self,
        image_source: Union[str, bytes],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """Complete test pipeline"""
        try:
            # Load image
            image = self.load_image(image_source)
            if not image:
                return {"error": "Failed to load image"}
            
            # Generate embedding
            embedding = self.generate_embedding(image)
            if not embedding:
                return {"error": "Failed to generate embedding"}
            
            # Perform search
            results = self.search_similar(embedding, top_k, filters)
            
            # Visualize results
            self.visualize_results(results, image)
            
            return results
            
        except Exception as e:
            logging.error(f"Test search failed: {str(e)}")
            return {"error": str(e)}

# Example Usage
if __name__ == "__main__":
    # Configuration
    PINECONE_API_KEY = "your-api-key"  # Replace with your actual key
    TEST_IMAGE = "path/to/your/test_image.jpg"  # Can be path or URL
    
    # Initialize tester
    tester = ImageSearchTester(api_key=PINECONE_API_KEY)
    
    # Example with filters (optional)
    filters = {
        "category": {"$eq": "furniture"},
        "price": {"$lt": 100}
    }
    
    # Run test
    results = tester.test_search(
        image_source=TEST_IMAGE,
        top_k=5,
        filters=None  # Remove or replace with actual filters
    )