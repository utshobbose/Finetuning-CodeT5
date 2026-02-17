"""
Inference script for fine-tuned CodeT5 with LoRA.
Loads base model + LoRA adapters.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Salesforce/codet5-large"
ADAPTER_PATH = "./codet5-lora-finetuned/final"


def load_model():
    """Load base model + LoRA adapters."""
    print("Loading base model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load base model in fp16 for faster inference
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    # Merge adapters for faster inference (optional)
    # model = model.merge_and_unload()
    
    print("Model ready!")
    return model, tokenizer


def generate_fix(model, tokenizer, buggy_code, max_length=512):
    """Generate bug explanation and fix."""
    input_text = f"fix bug: {buggy_code}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    model, tokenizer = load_model()
    
    # Test examples
    test_cases = [
        """def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)""",
        
        """def get_item(lst, index):
    return lst[index]

my_list = [1, 2, 3]
print(get_item(my_list, 5))""",
        
        """def add_numbers(a, b)
    return a + b

print(add_numbers(3, 4))""",
    ]
    
    for i, test_code in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print("Buggy code:")
        print(test_code)
        print(f"\n{'-'*60}")
        print("Model output:")
        fix = generate_fix(model, tokenizer, test_code)
        print(fix)
        print(f"{'='*60}")


if __name__ == "__main__":
    main()