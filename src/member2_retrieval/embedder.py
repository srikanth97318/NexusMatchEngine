import logging
import hashlib
from typing import Dict, List, Any, Optional
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
        if not FLAG_EMBEDDING_AVAILABLE:
            raise ImportError(
                "The 'FlagEmbedding' library is required to run the native BGEM3Embedder. Please install it using pip."
            )

        self.model_name = model_name or settings.BGE_MODEL_NAME
        self.use_fp16 = use_fp16
        self.dimension = 1024

        logger.info(f"Loading native BGE-M3 model: {self.model_name}...")
        try:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            logger.info("BGE-M3 Model loaded natively successfully.")
        except Exception as e:
            logger.critical(f"Could not load native BGE-M3 model: {e}")
            raise RuntimeError(f"BGE-M3 model loading failed: {e}") from e

    def generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate multi-vector embeddings for given text payloads.

        Returns:
            List of dicts containing dense vector representation,
            sparse tokens weights mapping, and late-interaction token tensors.
        """
        # Encode text using the native BGE-M3 model
        output = self.model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=True
        )

        results = []
        for i in range(len(texts)):
            dense = output["dense_vecs"][i].tolist()

            # Convert lexical weights dict to Qdrant sparse format: {indices: [int], values: [float]}
            sparse_dict = output["lexical_weights"][i]
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
                    token_id = (
                        int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 30522
                    )
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
            colbert = output["colbert_vecs"][i]
            if hasattr(colbert, "tolist"):
                colbert = colbert.tolist()

            results.append(
                {"dense": dense, "sparse": sparse, "late_interaction": colbert}
            )

        logger.info(f"Generated native BGE-M3 embeddings for {len(texts)} texts.")
        return results
