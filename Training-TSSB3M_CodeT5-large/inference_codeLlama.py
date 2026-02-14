"""
Inference script for fine-tuned CodeLlama model.
Use this to test the model after training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
FINETUNED_PATH = "./codellama-finetuned/final"


def load_model():
    """Load fine-tuned model."""
    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, FINETUNED_PATH)
    model.eval()
    
    return model, tokenizer


def generate_fix(model, tokenizer, buggy_code):
    """Generate bug explanation and fix."""
    prompt = f"""<s>[INST] <<SYS>>
You are an expert Python debugger. Analyze the buggy code, explain the bug in detail, and provide the corrected code.
<</SYS>>

Explain the bug in this code and provide the fix:

Buggy code:
```python
{buggy_code}
``` [/INST]
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response


def main():
    model, tokenizer = load_model()
    
    # Test example
    test_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""
    
    print("Buggy code:")
    print(test_code)
    print("\n" + "="*50 + "\n")
    
    fix = generate_fix(model, tokenizer, test_code)
    print("Model output:")
    print(fix)


if __name__ == "__main__":
    main()