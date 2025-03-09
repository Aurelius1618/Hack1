import os
import pinecone
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

# Define a HybridQuery class for improved hybrid search
class HybridQuery:
    """
    Class for hybrid search queries combining dense and sparse vectors
    """
    def __init__(self, sparse_vector: Dict[str, Union[List[int], List[float]]], 
                 dense_vector: List[float], alpha: float = 0.7):
        """
        Initialize a hybrid query
        
        Args:
            sparse_vector (Dict): Sparse vector with indices and values
            dense_vector (List[float]): Dense vector
            alpha (float): Weight between sparse and dense (0 = sparse only, 1 = dense only)
        """
        self.sparse_vector = sparse_vector
        self.dense_vector = dense_vector
        self.alpha = alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for Pinecone
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
            "alpha": self.alpha
        }

def initialize_pinecone():
    """
    Initialize Pinecone connection and create indexes if they don't exist
    """
    if not PINECONE_API_KEY:
        logger.error("Pinecone API key not found. Please set PINECONE_API_KEY in .env file.")
        return False
    
    try:
        # Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        # Define indexes to create
        indexes = {
            "bond_legal": {
                "dimension": 768,
                "metric": "dotproduct",
                "pods": 2,
                "replicas": 1
            },
            "bond_financial": {
                "dimension": 768,
                "metric": "cosine",
                "pods": 1,
                "replicas": 1
            }
        }
        
        # Create indexes if they don't exist
        existing_indexes = pinecone.list_indexes()
        
        for index_name, config in indexes.items():
            if index_name not in existing_indexes:
                logger.info(f"Creating index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=config["dimension"],
                    metric=config["metric"],
                    pods=config["pods"],
                    replicas=config["replicas"]
                )
                logger.info(f"Index {index_name} created successfully")
            else:
                logger.info(f"Index {index_name} already exists")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return False

def get_pinecone_index(index_name="bond_metadata"):
    """
    Get the Pinecone index for the application
    
    Args:
        index_name (str): Name of the index to get
        
    Returns:
        pinecone.Index: Pinecone index
    """
    # Initialize Pinecone if not already initialized
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # Connect to the index
    try:
        index = pinecone.Index(index_name)
        return index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index {index_name}: {str(e)}")
        return None

def hybrid_search(query, namespace="bond_metadata", top_k=5, alpha=0.7):
    """
    Perform hybrid search using both dense and sparse vectors
    
    Args:
        query (str): The search query
        namespace (str): The namespace to search in
        top_k (int): Number of results to return
        alpha (float): Weight for hybrid search (0 = sparse only, 1 = dense only)
        
    Returns:
        list: Search results
    """
    from app.utils.embeddings import get_dense_embedding, get_sparse_embedding
    
    # Get dense and sparse embeddings
    dense_vec = get_dense_embedding(query)
    sparse_vec = get_sparse_embedding(query)
    
    # Create hybrid query
    hybrid_query = HybridQuery(sparse_vec, dense_vec, alpha)
    
    # Get the index
    index = get_pinecone_index(namespace.split('/')[0])
    if not index:
        logger.error(f"Could not connect to index for namespace {namespace}")
        return []
    
    # Perform hybrid search
    try:
        results = index.query(
            **hybrid_query.to_dict(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        return results
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        
        # Fallback to dense search only
        try:
            results = index.query(
                vector=dense_vec,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            return results
        except Exception as e2:
            logger.error(f"Error performing fallback dense search: {str(e2)}")
            return []

# Define the namespaces for our different data types
NAMESPACES = {
    "BOND_METADATA": "bond_metadata/main",  # 768d financial embeddings
    "FINANCIAL_METRICS": "financial_metrics/main",  # 512d financial ratio vectors
    "CASHFLOW_PATTERNS": "cashflow_patterns/main"  # 256d time-series vectors
} 