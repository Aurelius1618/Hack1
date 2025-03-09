import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mistral_model():
    """
    Initialize Mistral-7B model with 4-bit quantization and LoRA adapters
    
    Returns:
        model: Quantized Mistral-7B model
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        import torch
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load the model with quantization
        model_id = "mistralai/Mistral-7B-v0.1"
        logger.info(f"Loading {model_id} with 4-bit quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure LoRA adapters for query classification
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA adapters
        logger.info("Applying LoRA adapters to the model...")
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer
    
    except ImportError as e:
        logger.error(f"Error importing required modules: {str(e)}")
        logger.warning("Falling back to simpler classification method")
        return None, None

def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify a query using Mistral-7B with LoRA adapters
    
    Args:
        query (str): User query
        
    Returns:
        Dict[str, Any]: Classification results
    """
    model, tokenizer = get_mistral_model()
    
    if model is None or tokenizer is None:
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
        
        # Generate classification
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Extract classification
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace(prompt, "").strip().lower()
        
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
            "reasoning": f"Classified using Mistral-7B with LoRA adapters: {result}"
        }
    
    except Exception as e:
        logger.error(f"Error classifying query with Mistral-7B: {str(e)}")
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
        Dict[str, Any]: Best reasoning path and result
    """
    model, tokenizer = get_mistral_model()
    
    if model is None or tokenizer is None:
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
        
        # Generate reasoning
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Extract reasoning
        reasoning = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = reasoning.replace(prompt, "").strip()
        
        thoughts.append({
            "strategy": strategy,
            "reasoning": reasoning
        })
    
    # Critic evaluation
    scores = []
    
    for thought in thoughts:
        # Create critic prompt
        critic_prompt = f"""
        Evaluate this reasoning for a bond-related query:
        
        Query: {query}
        
        Reasoning ({thought['strategy']}):
        {thought['reasoning']}
        
        Rate this reasoning from 0-10 based on:
        - Financial accuracy
        - Logical coherence
        - Relevance to query
        
        Rating (0-10):
        """
        
        # Generate critic score
        inputs = tokenizer(critic_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Extract score
        score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score_text = score_text.replace(critic_prompt, "").strip()
        
        # Parse score
        try:
            score = float(re.search(r'\d+(?:\.\d+)?', score_text).group(0))
        except:
            score = 5.0  # Default score
        
        scores.append(score)
    
    # Get best reasoning
    best_index = scores.index(max(scores))
    best_thought = thoughts[best_index]
    
    return {
        "reasoning": best_thought["reasoning"],
        "strategy": best_thought["strategy"],
        "score": scores[best_index]
    } 