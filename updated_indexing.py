import json
import pinecone
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
from datasketch import MinHash, MinHashLSH
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Optional
import unicodedata
import logging
from tqdm import tqdm

# Configuration
class Config:
    # Model Configuration
    MODEL_NAME = "openai/clip-vit-large-patch14"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME = "enhanced-image-index"
    PINECONE_METRIC = "cosine"
    PINECONE_DIMENSION = 768  # For CLIP-large
    
    # Processing Parameters
    BATCH_SIZE = 128
    AUGMENTATION_COUNT = 3
    DEDUP_THRESHOLD = 0.9
    HYBRID_WEIGHTS = {"image": 0.7, "text": 0.3}
    
    # Image Transformations
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("indexing.log"), logging.StreamHandler()]
)

class EnhancedImageIndexer:
    def __init__(self, config: Config):
        self.config = config
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize models, processors, and Pinecone index"""
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(self.config.MODEL_NAME).to(self.config.DEVICE)
        self.processor = CLIPProcessor.from_pretrained(self.config.MODEL_NAME)
        
        # Initialize image transformations
        self.base_transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.NORMALIZE_MEAN, self.config.NORMALIZE_STD)
        ])
        
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(self.config.IMAGE_SIZE, scale=(0.8, 1.0)),
            self.base_transform
        ])
        
        # Initialize Pinecone
        pinecone.init(api_key="your-pinecone-api-key")
        self._init_pinecone_index()
        
        # Initialize deduplication
        self.lsh = MinHashLSH(threshold=self.config.DEDUP_THRESHOLD, num_perm=128)
        
    def _init_pinecone_index(self):
        """Initialize or connect to Pinecone index"""
        if self.config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC
            )
        self.index = pinecone.Index(self.config.PINECONE_INDEX_NAME)
        
    @staticmethod
    def sanitize_id(text: str) -> str:
        """Normalize and sanitize ID strings"""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download and validate image"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logging.error(f"Image download failed: {url} - {str(e)}")
            return None
    
    def _generate_augmented_embeddings(self, image: Image.Image) -> List[List[float]]:
        """Generate multiple augmented embeddings"""
        embeddings = []
        with torch.no_grad():
            # Base embedding
            base_tensor = self.base_transform(image).unsqueeze(0).to(self.config.DEVICE)
            embeddings.append(self.model.get_image_features(base_tensor))
            
            # Augmented embeddings
            for _ in range(self.config.AUGMENTATION_COUNT):
                aug_tensor = self.aug_transform(image).unsqueeze(0).to(self.config.DEVICE)
                embeddings.append(self.model.get_image_features(aug_tensor))
                
        return [e.cpu().numpy().tolist()[0] for e in embeddings]
    
    def _create_minhash(self, embedding: List[float]) -> MinHash:
        """Create MinHash for deduplication"""
        minhash = MinHash(num_perm=128)
        for idx in np.array(embedding).argsort()[-100:]:  # Top 100 features
            minhash.update(str(idx).encode('utf8'))
        return minhash
    
    def process_batch(self, batch: List[Dict]) -> int:
        """Process a batch of records with error handling"""
        vectors = []
        successful = 0
        
        for record in batch:
            try:
                # Sanitize and validate
                record_id = self.sanitize_id(record["id"])
                image = self._download_image(record["image_url"])
                if not image:
                    continue
                
                # Generate embeddings
                embeddings = self._generate_augmented_embeddings(image)
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                
                # Check duplicates
                minhash = self._create_minhash(avg_embedding)
                if self.lsh.query(minhash):
                    logging.info(f"Skipping duplicate: {record_id}")
                    continue
                self.lsh.insert(record_id, minhash)
                
                # Generate text embedding
                text_embedding = self.model.get_text_features(
                    **self.processor(
                        text=record.get("description", ""), 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True
                    ).to(self.config.DEVICE)
                ).cpu().numpy().tolist()[0]
                
                # Create hybrid embedding
                hybrid_embedding = (
                    np.array(avg_embedding) * self.config.HYBRID_WEIGHTS["image"] +
                    np.array(text_embedding) * self.config.HYBRID_WEIGHTS["text"]
                ).tolist()
                
                # Prepare metadata
                metadata = {
                    k: v for k, v in record.items() 
                    if k not in ["id", "image_url", "description"]
                }
                
                vectors.append((record_id, hybrid_embedding, metadata))
                successful += 1
                
            except Exception as e:
                logging.error(f"Error processing {record.get('id')}: {str(e)}")
        
        # Batch upsert to Pinecone
        if vectors:
            self.index.upsert(vectors=vectors)
            
        return successful
    
    def index_from_jsonl(self, file_path: str):
        """Main indexing method"""
        with open(file_path, "r") as f:
            batch = []
            total = 0
            for line in tqdm(f, desc="Indexing"):
                try:
                    record = json.loads(line)
                    batch.append(record)
                    if len(batch) >= self.config.BATCH_SIZE:
                        total += self.process_batch(batch)
                        batch = []
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON: {line.strip()}")
            
            if batch:
                total += self.process_batch(batch)
                
        logging.info(f"Indexing complete. Total records: {total}")

class QueryProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.model = CLIPModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
        self.processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
        self.index = pinecone.Index(config.PINECONE_INDEX_NAME)
        
    def query_image(self, image: Image.Image, top_k: int = 10):
        """Enhanced query with augmentation"""
        # Generate query embeddings
        query_embeddings = []
        transforms = [
            self.base_transform,
            self.aug_transform,
            transforms.functional.rotate(15),
            transforms.functional.hflip,
        ]
        
        for transform in transforms:
            try:
                tensor = transform(image).unsqueeze(0).to(self.config.DEVICE)
                with torch.no_grad():
                    query_embeddings.append(self.model.get_image_features(tensor))
            except Exception as e:
                logging.error(f"Transform failed: {str(e)}")
                
        avg_embedding = torch.mean(torch.stack(query_embeddings), dim=0)
        return self.index.query(queries=[avg_embedding.cpu().numpy().tolist()], top_k=top_k)

# Usage Example
if __name__ == "__main__":
    config = Config()
    
    # Indexing
    indexer = EnhancedImageIndexer(config)
    indexer.index_from_jsonl("./jsonl_output/pins_details.jsonl")
    
    # Querying
    query_processor = QueryProcessor(config)
    sample_image = Image.open("query_image.jpg")
    results = query_processor.query_image(sample_image)
    print("Search results:", results)