import torch
import json
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an expert agricultural advisor with deep knowledge of plant diseases, "
    "organic and chemical treatments, and regional farming practices across India. "
    "Always respond with a valid JSON object following the specified schema."
)

def format_inference_prompt(instruction, context=""):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n\nContext: {context}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def main():
    print("Loading config...")
    with open("configs/llm_config.yaml") as f:
        cfg = yaml.safe_load(f)

    base_model_name = cfg["base_model"]
    adapter_path = "models/llm/qwen2.5_3b_qlora_adapter"
    
    print(f"Loading Base Model: {base_model_name}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA Adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_cases = [
        {"instruction": "Identify the symptoms of Late Blight in tomato.", "input": "Tomato plant with dark lesions."},
        {"instruction": "Recommend an organic treatment for Apple Scab.", "input": "Apple crops."},
        {"instruction": "What is the best chemical treatment for Potato Early Blight?", "input": "Potato farm in Gujarat."},
        {"instruction": "How to prevent Grape Black Rot?", "input": "Grape vineyard."},
        {"instruction": "Provide treatment for Common Rust in corn.", "input": "Corn crop infected with Common Rust."},
        {"instruction": "What are the common symptoms of Peach Bacterial Spot?", "input": "Peach trees."},
        {"instruction": "How to identify and control Citrus Greening?", "input": "Citrus farm with Huanglongbing disease."},
        {"instruction": "Suggest organic options for Strawberry leaf scorch.", "input": "Strawberry farming."},
        {"instruction": "How do I identify Bacterial Spot on Bell Peppers?", "input": "Pepper bell crops."},
        {"instruction": "What causes Powdery Mildew in cherry trees and how to prevent it?", "input": "Cherry Powdery mildew."}
    ]

    print("\n" + "="*80)
    print("--- Starting Inference on 10 Test Cases ---")
    print("="*80 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/10] Test Case")
        print(f"Instruction: {test['instruction']}")
        print(f"Context: {test['input']}\n")
        
        prompt = format_inference_prompt(test["instruction"], test["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7, # Lower temperature for formatting
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Response:\n{response}")
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()
