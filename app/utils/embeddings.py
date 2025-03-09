import os
import numpy as np
from typing import List, Dict, Union
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize the embedding models
def get_embedding_model():
    """
    Get the embedding model for dense vectors
    """
    # Using a financial domain-specific model if available, otherwise use a general model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
    
    # In production, we would use a financial domain-specific model like:
    # model_name = "yiyanghkust/finbert-tone"
    
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def get_bge_embedding_model():
    """
    Get the BGE embedding model for improved dense vectors
    """
    model_name = "BAAI/bge-small-en-v1.5"
    
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

# Initialize BM25 for sparse embeddings
class BM25Encoder:
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.stop_words = set(stopwords.words('english'))
        
    def fit(self, corpus: List[str]):
        """
        Fit the BM25 model on a corpus
        
        Args:
            corpus (List[str]): List of documents
        """
        tokenized_corpus = []
        for doc in corpus:
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            tokenized_corpus.append(tokens)
            
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def encode(self, query: str) -> Dict[str, Union[List[int], List[float]]]:
        """
        Encode a query into a sparse vector
        
        Args:
            query (str): The query to encode
            
        Returns:
            Dict: Sparse vector representation with indices and values
        """
        if self.bm25 is None:
            raise ValueError("BM25 model not fitted. Call fit() first.")
            
        # Tokenize the query
        tokens = word_tokenize(query.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Get the scores for each token in the corpus
        token_scores = {}
        for token in tokens:
            if token in self.bm25.idf:
                token_scores[token] = self.bm25.idf[token]
        
        # Create sparse vector
        indices = []
        values = []
        
        # Get unique tokens in the corpus
        all_tokens = set()
        for doc in self.tokenized_corpus:
            all_tokens.update(doc)
        
        # Create a mapping from token to index
        token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        
        # Fill the sparse vector
        for token, score in token_scores.items():
            if token in token_to_idx:
                indices.append(token_to_idx[token])
                values.append(score)
        
        return {"indices": indices, "values": values}

# Global instances
_dense_embeddings = None
_bge_embeddings = None
_bm25_encoder = None

def get_dense_embedding(text: str) -> List[float]:
    """
    Get dense embedding for a text
    
    Args:
        text (str): The text to embed
        
    Returns:
        List[float]: Dense vector embedding
    """
    global _dense_embeddings
    
    if _dense_embeddings is None:
        _dense_embeddings = get_embedding_model()
    
    # Get the embedding
    embedding = _dense_embeddings.embed_query(text)
    
    return embedding

def get_bge_embedding(text: str) -> List[float]:
    """
    Get BGE embedding for a text
    
    Args:
        text (str): The text to embed
        
    Returns:
        List[float]: BGE vector embedding
    """
    global _bge_embeddings
    
    if _bge_embeddings is None:
        _bge_embeddings = get_bge_embedding_model()
    
    # Get the embedding
    embedding = _bge_embeddings.embed_query(text)
    
    return embedding

def get_sparse_embedding(text: str) -> Dict[str, Union[List[int], List[float]]]:
    """
    Get sparse embedding for a text
    
    Args:
        text (str): The text to embed
        
    Returns:
        Dict: Sparse vector representation
    """
    global _bm25_encoder
    
    if _bm25_encoder is None:
        _bm25_encoder = BM25Encoder()
        # We need to fit the BM25 model on a corpus
        # For now, we'll use a dummy corpus, but in production
        # this would be fitted on the actual bond data
        dummy_corpus = [
            "bond yield maturity interest rate",
            "corporate bond government bond municipal bond",
            "ISIN code identifier security",
            "cash flow payment schedule coupon",
            "credit rating default risk"
        ]
        _bm25_encoder.fit(dummy_corpus)
    
    # Get the sparse embedding
    sparse_embedding = _bm25_encoder.encode(text)
    
    return sparse_embedding

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks for processing
    
    Args:
        text (str): The text to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    return chunks 