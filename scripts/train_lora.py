#!/usr/bin/env python
"""
Script to train LoRA adapters on bond-specific data
"""

import os
import logging
import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters on bond-specific data")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model ID")
    parser.add_argument("--dataset_path", type=str, default="data/bond_queries.json", help="Path to dataset file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for trained model")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    return parser.parse_args()

class BondDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.inputs = []
        for item in data:
            query = item.get("query", "")
            response = item.get("response", "")
            prompt = f"Query: {query}\n\nResponse:"
            
            # Tokenize inputs
            tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", 
                                       max_length=512, return_tensors="pt")
            tokenized_response = tokenizer(response, truncation=True, padding="max_length",
                                         max_length=512, return_tensors="pt")
            
            # Combine for causal LM training
            input_ids = torch.cat([tokenized_prompt["input_ids"], tokenized_response["input_ids"][:, 1:]], dim=1)
            attention_mask = torch.cat([tokenized_prompt["attention_mask"], tokenized_response["attention_mask"][:, 1:]], dim=1)
            
            self.inputs.append({
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "labels": input_ids[0].clone()  # Use input_ids as labels for causal LM
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

def main():
    args = parse_args()
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = device
    
    # Set appropriate dtype based on device
    if device == "cuda":
        torch_dtype = torch.float16  # Use half precision for GPU to save memory
        logger.info(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
    else:
        torch_dtype = torch.float32
        logger.warning("GPU not available. Falling back to CPU, which will be significantly slower")
    
    # Load the model
    logger.info(f"Loading {args.model_id} on {device.upper()}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set pad_token_id in the model's generation config to avoid warnings
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA adapters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA adapters
    logger.info("Applying LoRA adapters to the model...")
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    logger.info(f"Loading bond-specific dataset from {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    
    # Create dataset
    train_dataset = BondDataset(data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # Start training
    logger.info("Starting fine-tuning on bond-specific data")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning completed successfully")

if __name__ == "__main__":
    main() 