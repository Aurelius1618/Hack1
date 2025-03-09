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
            "screener": "screener"
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