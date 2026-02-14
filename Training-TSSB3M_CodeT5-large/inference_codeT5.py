"""
Inference script for fine-tuned CodeT5 model.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "./codet5-finetuned/final"


def load_model():
    """Load fine-tuned CodeT5 model."""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    return model, tokenizer


def generate_fix(model, tokenizer, buggy_code):
    """Generate bug explanation and fix."""
    # T5 uses task prefix
    input_text = f"fix bug: {buggy_code}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,           # Better quality than sampling
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    model, tokenizer = load_model()
    
    # Test example
    test_code = """def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)"""
    
    print("Buggy code:")
    print(test_code)
    print("\n" + "="*50 + "\n")
    
    fix = generate_fix(model, tokenizer, test_code)
    print("Model output:")
    print(fix)


if __name__ == "__main__":
    main()