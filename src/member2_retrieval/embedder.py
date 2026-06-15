import logging
import hashlib
from typing import Dict, List, Any, Optional
import numpy as np
from src.config import settings

try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """BGE-M3 multi-vector embedding engine supporting Dense, Sparse, & Late-interaction tensors."""
    
    def __init__(self, model_name: Optional[str] = None, use_fp16: bool = False):
        self.model_name = model_name or settings.BGE_MODEL_NAME
        self.use_fp16 = use_fp16
        self.dimension = 1024
        self.model = None
        
        if FLAG_EMBEDDING_AVAILABLE:
            try:
                logger.info(f"Attempting to load SOTA BGE-M3 model: {self.model_name}...")
                self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
                logger.info("BGE-M3 Model loaded natively successfully.")
            except Exception as e:
                logger.warning(f"Could not load native BGE-M3 model: {e}. Activating deterministic simulation fallback.")
        else:
            logger.warning("FlagEmbedding is not installed in this environment. Activating deterministic simulation fallback.")
            
    def generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate multi-vector embeddings for given text payloads.
        
        Returns:
            List of dicts containing dense vector representation,
            sparse tokens weights mapping, and late-interaction token tensors.
        """
        if self.model is not None:
            try:
                # Encode text using the native BGE-M3 model
                output = self.model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
                
                results = []
                for i in range(len(texts)):
                    dense = output['dense_vecs'][i].tolist()
                    
                    # Convert lexical weights dict to Qdrant sparse format: {indices: [int], values: [float]}
                    sparse_dict = output['lexical_weights'][i]
                    indices = []
                    values = []
                    
                    for token, weight in sparse_dict.items():
                        try:
                            # If token is a string representation of ID, cast to int
                            token_id = int(token)
                            indices.append(token_id)
                            values.append(float(weight))
                        except ValueError:
                            # Hash the token string to get a deterministic index within the BGE-M3 vocabulary limits
                            token_id = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % 30522
                            indices.append(token_id)
                            values.append(float(weight))
                    
                    # Sort by indices for Qdrant validation requirements
                    if indices:
                        sorted_pairs = sorted(zip(indices, values))
                        indices, values = zip(*sorted_pairs)
                        sparse = {"indices": list(indices), "values": list(values)}
                    else:
                        sparse = {"indices": [101], "values": [1.0]}
                    
                    # Parse late-interaction embeddings (ColBERT vectors)
                    colbert = output['colbert_vecs'][i]
                    if hasattr(colbert, "tolist"):
                        colbert = colbert.tolist()
                        
                    results.append({
                        "dense": dense,
                        "sparse": sparse,
                        "late_interaction": colbert
                    })
                
                logger.info(f"Generated native BGE-M3 embeddings for {len(texts)} texts.")
                return results
            except Exception as e:
                logger.error(f"Error executing native BGE-M3 encoding: {e}. Falling back to simulation.")
                
        # Deterministic simulation fallback
        results = []
        for text in texts:
            # Hash text to seed random generator for deterministic output
            hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(hash_val)
            
            # Generate deterministic dense vector
            dense_vector = rng.uniform(-0.1, 0.1, self.dimension).tolist()
            
            # Generate deterministic sparse vector based on word vocabulary mapping
            words = [w.lower() for w in text.split() if w.isalnum()]
            indices = []
            values = []
            for w in set(words):
                token_id = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % 30000
                weight = float(words.count(w) / len(words))
                indices.append(token_id)
                values.append(weight)
                
            if indices:
                sorted_pairs = sorted(zip(indices, values))
                indices, values = zip(*sorted_pairs)
                sparse_vector = {"indices": list(indices), "values": list(values)}
            else:
                sparse_vector = {"indices": [101], "values": [1.0]}
                
            # Generate deterministic late-interaction tensors
            colbert_vectors = [rng.uniform(-0.1, 0.1, self.dimension).tolist() for _ in range(min(len(words), 8) or 1)]
            
            results.append({
                "dense": dense_vector,
                "sparse": sparse_vector,
                "late_interaction": colbert_vectors
            })
            
        logger.info(f"Generated simulated BGE-M3 embeddings for {len(texts)} texts.")
        return results
