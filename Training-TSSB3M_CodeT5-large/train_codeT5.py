"""
CodeT5-Large with LoRA Fine-Tuning
GPU: RTX 3060 12GB
Model: Salesforce/codet5-large
Method: LoRA (Low-Rank Adaptation) - only ~1% parameters trainable
Training Time: 2-3 hours
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
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ==============================
# Configuration
# ==============================

MODEL_NAME = "Salesforce/codet5-large"
DATA_PATH = "./processed_data"
OUTPUT_DIR = "./codet5-lora-finetuned"

MAX_SOURCE_LENGTH = 256    # Change from 512
MAX_TARGET_LENGTH = 256    # Change from 512

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,                    # Rank - higher = more capacity
    lora_alpha=32,          # Scaling factor
    target_modules=[        # Which layers to adapt
        "q",                # Query projection
        "v",                # Value projection
        "k",                # Key projection
        "o",                # Output projection
        "wi",               # Feed-forward input
        "wo",               # Feed-forward output
    ],
    lora_dropout=0.1,       # Dropout for regularization
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)


# ==============================
# Data Loading
# ==============================

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(tokenizer):
    print("Loading training data...")

    train_data = load_jsonl(f"{DATA_PATH}/train.jsonl")
    val_data = load_jsonl(f"{DATA_PATH}/val.jsonl")

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    def format_t5(example):
        return {
            "input": f"fix bug: {example['input']}",
            "target": example["output"]
        }

    train_formatted = [format_t5(ex) for ex in train_data]
    val_formatted = [format_t5(ex) for ex in val_data]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

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


# ==============================
# Main Training
# ==============================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Model: {MODEL_NAME}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32  # Use full precision for stability
    )

    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    # model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_dataset, val_dataset = prepare_dataset(tokenizer)

    # ==============================
    # Training Arguments
    # ==============================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training
        num_train_epochs=3,
        learning_rate=1e-4,        # Higher LR works with LoRA
        weight_decay=0.01,
        label_smoothing_factor=0.1,

        # Batch - can use larger batch with LoRA
        per_device_train_batch_size=2,      # Change from 4
        per_device_eval_batch_size=2,       # Change from 4
        gradient_accumulation_steps=4,      # Change from 2 (effective batch still 8)

        # Stability
        fp16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.1,          # Longer warmup
        lr_scheduler_type="cosine",

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,

        # Performance
        dataloader_num_workers=2,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving final model...")
    # Save only LoRA adapters (small file)
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("Training complete!")


if __name__ == "__main__":
    main()