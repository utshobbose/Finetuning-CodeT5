"""
CodeLlama Fine-tuning for Bug Fix Explanation and Correction
Based on thesis: "AI-Assisted Code Generation Tools: A New Frontier in Software Development"

Hardware: RTX 3060 12GB
Model: codellama/CodeLlama-7b-Instruct-hf
Method: QLoRA (4-bit quantization)
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
DATA_PATH = "./processed_data"
OUTPUT_DIR = "./codellama-finetuned"
MAX_SEQ_LENGTH = 2048

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_instruction(sample):
    """Format data into CodeLlama chat template."""
    prompt = f"""<s>[INST] <<SYS>>
You are an expert Python debugger. Analyze the buggy code, explain the bug in detail, and provide the corrected code.
<</SYS>>

{sample['instruction']}

Buggy code:
```python
{sample['input']}
``` [/INST]

{sample['output']}</s>"""
    
    return {"text": prompt}


def tokenize_function(examples, tokenizer):
    """Tokenize the texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )


def prepare_dataset(tokenizer):
    """Load and prepare training data."""
    print("Loading training data...")
    
    train_data = load_jsonl(f"{DATA_PATH}/train.jsonl")
    val_data = load_jsonl(f"{DATA_PATH}/val.jsonl")
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(format_instruction)
    val_dataset = val_dataset.map(format_instruction)
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Need GPU for training.")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=1000,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_steps=100,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        save_total_limit=2,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("Training complete!")


if __name__ == "__main__":
    main()