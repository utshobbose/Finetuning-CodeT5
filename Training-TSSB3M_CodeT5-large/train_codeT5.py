"""
CodeT5 Fine-tuning for Bug Fix Explanation and Correction
Based on thesis: "AI-Assisted Code Generation Tools: A New Frontier in Software Development"

Hardware: RTX 3060 12GB
Model: Salesforce/codet5-large
Method: Full fine-tuning (no quantization needed)
Training Time: 8-12 hours
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_NAME = "Salesforce/codet5-large"  # 770M params
DATA_PATH = "./processed_data"
OUTPUT_DIR = "./codet5-finetuned"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

# LoRA Configuration (optional - remove for full fine-tuning)
USE_LORA = False  # Set True for faster training, False for full fine-tuning

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v", "k", "o", "wi", "wo"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(tokenizer):
    """Load and prepare training data for T5 format."""
    print("Loading training data...")
    
    train_data = load_jsonl(f"{DATA_PATH}/train.jsonl")
    val_data = load_jsonl(f"{DATA_PATH}/val.jsonl")
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")
    
    def format_t5(example):
        # T5 uses task prefix format
        input_text = f"fix bug: {example['input']}"
        target_text = example['output']
        return {"input": input_text, "target": target_text}
    
    train_formatted = [format_t5(ex) for ex in train_data]
    val_formatted = [format_t5(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    def preprocess_function(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding="max_length",
        )
        
        targets = tokenizer(
            examples["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        
        # Replace padding token id with -100 for loss computation
        labels = targets["input_ids"].copy()
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in labels
        ]
        
        inputs["labels"] = labels
        return inputs
    
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    return train_dataset, val_dataset


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Need GPU for training.")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print(f"Model: {MODEL_NAME}")
    
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use fp16 for faster training
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    
    if USE_LORA:
        print("Applying LoRA...")
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()
    else:
        print("Full fine-tuning (all parameters)")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    # Training arguments for RTX 3060 12GB
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=1000,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=False,
        max_grad_norm=1.0,
        warmup_steps=500,
        lr_scheduler_type="linear",
        report_to="tensorboard",
        eval_strategy="steps",        # Fixed here
        eval_steps=500,
        load_best_model_at_end=True,
        save_total_limit=2,
        dataloader_num_workers=2,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
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
    
    print(f"\nSaving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("Training complete!")


if __name__ == "__main__":
    main()