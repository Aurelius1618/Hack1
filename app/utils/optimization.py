import os
import logging
from typing import Dict, Any, List, Callable, Optional
from functools import lru_cache
import dask.dataframe as dd
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache for ISIN lookups
ISIN_CACHE = {}
CACHE_TTL = 3600  # 1 hour in seconds

def cache_result(key: str, result: Any, ttl: int = CACHE_TTL) -> None:
    """
    Cache a result with a TTL
    
    Args:
        key (str): Cache key
        result (Any): Result to cache
        ttl (int): Time to live in seconds
    """
    ISIN_CACHE[key] = {
        "result": result,
        "timestamp": time.time(),
        "ttl": ttl
    }
    logger.debug(f"Cached result for key {key}")

def get_cached_result(key: str) -> Optional[Any]:
    """
    Get a cached result if it exists and is not expired
    
    Args:
        key (str): Cache key
        
    Returns:
        Optional[Any]: Cached result or None if not found or expired
    """
    if key not in ISIN_CACHE:
        return None
    
    cache_entry = ISIN_CACHE[key]
    current_time = time.time()
    
    if current_time - cache_entry["timestamp"] > cache_entry["ttl"]:
        # Cache entry has expired
        del ISIN_CACHE[key]
        return None
    
    logger.debug(f"Cache hit for key {key}")
    return cache_entry["result"]

@lru_cache(maxsize=128)
def cached_isin_lookup(isin: str) -> Dict[str, Any]:
    """
    Cached lookup for ISIN details
    
    Args:
        isin (str): ISIN code
        
    Returns:
        Dict[str, Any]: Bond details
    """
    # This is a placeholder for the actual lookup function
    # In a real implementation, this would call the data processor
    from app.data.data_processor import FinancialDataProcessor
    
    data_processor = FinancialDataProcessor()
    data_processor.load_data()
    
    return data_processor.get_bond_details(isin)

def parallelize_cashflow_calculations(func: Callable, isins: List[str], *args, **kwargs) -> Dict[str, Any]:
    """
    Parallelize cash flow calculations using Dask
    
    Args:
        func (Callable): Function to parallelize
        isins (List[str]): List of ISINs to process
        *args: Additional arguments for the function
        **kwargs: Additional keyword arguments for the function
        
    Returns:
        Dict[str, Any]: Results for each ISIN
    """
    import pandas as pd
    
    # Create a DataFrame with ISINs
    df = pd.DataFrame({"isin": isins})
    
    # Convert to Dask DataFrame for parallel processing
    ddf = dd.from_pandas(df, npartitions=min(len(isins), os.cpu_count() or 4))
    
    # Apply the function to each ISIN
    results = ddf.apply(
        lambda row: func(row["isin"], *args, **kwargs),
        axis=1,
        meta=("result", "object")
    ).compute()
    
    # Convert results to dictionary
    return {isin: result for isin, result in zip(isins, results)}

def monitor_query_latency(func: Callable) -> Callable:
    """
    Decorator to monitor query latency
    
    Args:
        func (Callable): Function to monitor
        
    Returns:
        Callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.info(f"Query latency: {latency:.2f}ms")
        
        # Log warning if latency is too high
        if latency > 150:
            logger.warning(f"High query latency: {latency:.2f}ms")
        
        return result
    
    return wrapper 