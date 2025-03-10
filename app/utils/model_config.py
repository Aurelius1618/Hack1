import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import torch

# Load environment variables with explicit path
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
re.compile('<title>(.*)</title>')

def get_mistral_model():
    """
    Initialize Lamini finetuned model
    
    Returns:
        model: Lamini model instance
    """
    try:
        import lamini
        
        # Get API key from environment variables
        api_key = os.getenv("LAMINI_API_KEY")
        model_name = os.getenv("FINETUNED_MODEL_ID", "6fb049b7ef052adfa2a92125f0396fd307096f48a16b512a61861c2b265f1c5d")
        
        if not api_key:
            logger.error("LAMINI_API_KEY not found in environment variables")
            return None
            
        if not model_name:
            logger.error("FINETUNED_MODEL_ID not found in environment variables")
            return None
        
        logger.info(f"Initializing Lamini model: {model_name}")
        
        # Initialize Lamini model
        model = lamini.Lamini(
            model_name=model_name,
            api_key=api_key
        )
        
        return model
    
    except ImportError as e:
        logger.error(f"Error importing Lamini module: {str(e)}")
        logger.warning("Falling back to simpler classification method")
        return None

def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify a query using Lamini finetuned model
    
    Args:
        query (str): User query
        
    Returns:
        Dict[str, Any]: Classification results
    """
    model = get_mistral_model()
    
    if model is None:
        # Fallback to keyword-based classification
        return fallback_classify_query(query)
    
    try:
        # Prepare prompt for classification
        prompt = f"""
        Classify the following bond-related query into one of these categories:
        - directory: Queries about specific bonds or ISINs
        - finder: Queries about comparing yields or finding bonds with specific returns
        - cashflow: Queries about cash flow schedules or payment details
        - screener: Queries about screening bonds or analyzing financial health
        - yield_calculator: Queries about calculating yield to maturity, bond prices, or duration
        
        Query: {query}
        
        Category:
        """
        
        # Generate classification using Lamini
        result = model.generate(prompt)
        result = result.strip().lower()
        
        # Map to valid categories
        category_map = {
            "directory": "directory",
            "finder": "finder",
            "cashflow": "cashflow",
            "screener": "screener",
            "yield_calculator": "yield_calculator"
        }
        
        # Get the closest matching category
        category = None
        for key in category_map:
            if key in result:
                category = category_map[key]
                break
        
        if category is None:
            category = "screener"  # Default to screener if no match
        
        return {
            "next_node": category,
            "confidence": 0.92,
            "reasoning": f"Classified using Lamini finetuned model: {result}"
        }
    
    except Exception as e:
        logger.error(f"Error classifying query with Lamini: {str(e)}")
        return fallback_classify_query(query)

def fallback_classify_query(query: str) -> Dict[str, Any]:
    """
    Fallback classification using keyword matching
    
    Args:
        query (str): User query
        
    Returns:
        Dict[str, Any]: Classification results
    """
    import re
    
    # Extract ISIN if present
    isin_match = re.search(r'INE[A-Z0-9]{10}', query)
    
    if "ISIN" in query or "isin" in query or isin_match:
        return {
            "next_node": "directory",
            "confidence": 0.92,
            "reasoning": "Query contains ISIN code or explicitly asks for bond details"
        }
    elif any(keyword in query.lower() for keyword in ["ytm", "yield to maturity", "clean price", "dirty price", "duration", "macaulay", "modified duration"]):
        return {
            "next_node": "yield_calculator",
            "confidence": 0.90,
            "reasoning": "Query is about calculating yield to maturity, bond prices, or duration"
        }
    elif any(keyword in query.lower() for keyword in ["yield", "compare", "best", "highest", "return", "apy", "interest"]):
        return {
            "next_node": "finder",
            "confidence": 0.87,
            "reasoning": "Query is about comparing yields or finding bonds with specific returns"
        }
    elif any(keyword in query.lower() for keyword in ["cash flow", "cashflow", "payment", "schedule", "interest payment"]):
        return {
            "next_node": "cashflow",
            "confidence": 0.89,
            "reasoning": "Query is about cash flow schedules or payment details"
        }
    else:
        return {
            "next_node": "screener",
            "confidence": 0.85,
            "reasoning": "Query is about screening bonds or analyzing financial health"
        }

def actor_critic_flow(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Implement SQUID SQUAD's Contrastive CoT Methodology for financial reasoning
    
    Args:
        query (str): User query
        context (Dict[str, Any]): Additional context
        
    Returns:
        Dict[str, Any]: Result with reasoning
    """
    # Get Lamini model
    model = get_mistral_model()
    
    if model is None:
        # Fallback to simpler reasoning
        return {
            "reasoning": f"Direct answer to: {query}",
            "result": None
        }
    
    # Define reasoning strategies
    strategies = ['financial', 'legal', 'operational']
    
    # Generate multiple reasoning paths
    thoughts = []
    
    for strategy in strategies:
        # Create prompt for this strategy
        prompt = f"""
        Think through this bond-related query using {strategy} reasoning:
        
        Query: {query}
        
        Step-by-step {strategy} analysis:
        1. 
        """
        
        # Generate reasoning using Lamini
        try:
            result = model.generate(prompt)
            thoughts.append(f"{strategy.capitalize()} reasoning: {result}")
        except Exception as e:
            logger.error(f"Error generating {strategy} reasoning: {str(e)}")
            thoughts.append(f"{strategy.capitalize()} reasoning: Error occurred")
    
    # Combine thoughts
    combined_reasoning = "\n\n".join(thoughts)
    
    # Generate final answer
    final_prompt = f"""
    Based on the following analysis:
    
    {combined_reasoning}
    
    Provide a final answer to the query: {query}
    """
    
    try:
        final_answer = model.generate(final_prompt)
    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}")
        final_answer = "Unable to generate a response due to an error."
    
    return {
        "reasoning": combined_reasoning,
        "result": final_answer
    }

def handle_lamini_errors(func):
    """
    Decorator to handle Lamini API errors
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Lamini API Error: {str(e)}")
            return fallback_response()
    return wrapper

def fallback_response():
    """
    Generate a fallback response when Lamini API fails
    
    Returns:
        Dict[str, Any]: Fallback response
    """
    return {
        "status": "error",
        "message": "Unable to process your request at this time. Please try again later.",
        "data": None
    }

def batch_process(queries):
    """
    Process multiple queries in batch using Lamini
    
    Args:
        queries (List[str]): List of queries to process
        
    Returns:
        List[str]: List of responses
    """
    model = get_mistral_model()
    
    if model is None:
        return ["Unable to process query" for _ in queries]
    
    try:
        # Format queries using the template
        query_template = """Query: {user_input}\n\nResponse:"""
        formatted_queries = [query_template.format(user_input=q) for q in queries]
        
        # Generate responses in batch
        return model.generate_batch(formatted_queries)
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return ["Error processing query" for _ in queries]

# Cache for frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(query):
    """
    Cached version of generate function
    
    Args:
        query (str): Query to generate response for
        
    Returns:
        str: Generated response
    """
    model = get_mistral_model()
    
    if model is None:
        return "Unable to process query"
    
    try:
        return model.generate(query)
    except Exception as e:
        logger.error(f"Error in cached generate: {str(e)}")
        return "Error processing query" 