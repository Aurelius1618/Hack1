# LoRA Fine-Tuning for Bond Analysis

This guide explains how to fine-tune the Mistral-7B model with LoRA adapters for bond-specific tasks.

## Overview

Low-Rank Adaptation (LoRA) is a technique that significantly reduces the number of trainable parameters by adding low-rank matrices to specific layers of the model. This makes fine-tuning large language models more efficient while maintaining performance.

Our implementation uses LoRA to adapt Mistral-7B for bond-related queries, including:
- Yield to Maturity (YTM) calculations
- Bond pricing
- Duration and convexity analysis
- Cash flow analysis
- Bond comparison

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning) library
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The training dataset (`data/bond_queries.json`) contains pairs of bond-related queries and their corresponding responses. The format is:

```json
[
  {
    "query": "What is the YTM for INE08XP07258?",
    "response": "The Yield to Maturity (YTM) for INE08XP07258 is 7.85%..."
  },
  ...
]
```

You can extend this dataset with additional query-response pairs to improve the model's performance on specific bond-related tasks.

## Training

To train the model with LoRA adapters, run:

```bash
python scripts/train_lora.py --model_id mistralai/Mistral-7B-v0.1 --dataset_path data/bond_queries.json
```

### Parameters

- `--model_id`: Base model ID (default: "mistralai/Mistral-7B-v0.1")
- `--dataset_path`: Path to dataset file (default: "data/bond_queries.json")
- `--output_dir`: Output directory for trained model (default: "./results")
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of epochs (default: 3)

## LoRA Configuration

Our LoRA configuration targets the attention modules of the Mistral-7B model:

```python
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

- `r=16`: Rank of the low-rank matrices
- `lora_alpha=32`: Scaling factor for the LoRA updates
- `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`: Attention modules to apply LoRA to
- `lora_dropout=0.05`: Dropout rate for LoRA layers
- `bias="none"`: No bias parameters are trained
- `task_type="CAUSAL_LM"`: Task type for causal language modeling

## Using the Fine-Tuned Model

After training, the model and tokenizer will be saved to the specified output directory. You can load them with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load the LoRA adapters
model = PeftModel.from_pretrained(model, "./results")

# Use the model
prompt = "What is the YTM for INE08XP07258?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance Considerations

- Training on a GPU is highly recommended. CPU training will be extremely slow.
- For 8-bit quantization to save memory, add `load_in_8bit=True` when loading the base model.
- Adjust batch size based on your GPU memory. If you encounter out-of-memory errors, reduce the batch size or use gradient accumulation.

## Troubleshooting

- **Out of memory errors**: Reduce batch size, use gradient accumulation, or enable 8-bit quantization.
- **Slow training**: Ensure you're using a GPU. Consider using a smaller model or reducing the dataset size.
- **Poor performance**: Increase the dataset size, adjust learning rate, or try different LoRA configurations.

## References

- [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mistral-7B Model](https://huggingface.co/mistralai/Mistral-7B-v0.1) 