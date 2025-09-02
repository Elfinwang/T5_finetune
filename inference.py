import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import json
import pandas as pd
from tqdm import tqdm
import os

def load_model_and_tokenizer(base_model_path, lora_adapter_path):

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 加载LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    model.eval()
    
    return model, tokenizer

def inference(model, tokenizer, instruction, input_text="", max_length=64):

    if input_text:
        full_input = f"{instruction}\n{input_text}"
    else:
        full_input = instruction
    
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def load_test_data(test_file_path):

    with open(test_file_path, 'r', encoding='utf-8') as f:
        if test_file_path.endswith('.json'):
            test_data = json.load(f)
        else:
            test_data = [json.loads(line) for line in f]
    
    return test_data

def evaluate_test_set(model, tokenizer, test_data, output_csv_path):

    results = []
    
    print(f"Processing {len(test_data)} test examples...")
    
    for i, example in enumerate(tqdm(test_data, desc="Processing")):
        try:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            truth = example.get("output", "")
            
            # 生成预测结果
            prediction = inference(model, tokenizer, instruction, input_text)
            
            results.append({
                "input": input_text,
                "truth": truth,
                "prediction": prediction
            })
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "input": example.get("input", ""),
                "truth": example.get("output", ""),
                "prediction": "ERROR"
            })
    
    # 保存到CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_csv_path}")
    
    return results

def main():
    # 路径
    base_model_path = "./model/t5-small"
    lora_adapter_path = "./finetune_lora/t5-small-stats-v1/checkpoint-600"
    test_file_path = "./data/test_stats.json"
    output_csv_path = "./results/test_t5_small_ft_results.csv"

    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else '.', exist_ok=True)
    
    print("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_adapter_path)
    print("Model loaded successfully")
    
    print(f"Loading test data from {test_file_path}")
    test_data = load_test_data(test_file_path)
    print(f"Loaded {len(test_data)} test examples")
    
    print("Evaluating test set")
    results = evaluate_test_set(model, tokenizer, test_data, output_csv_path)
    
    print("\n=== Sample ===")
    for i, result in enumerate(results[:3]):  # 显示前3个结果
        print(f"\nExample {i+1}:")
        print(f"  Input: {result['input']}")
        print(f"  Truth: {result['truth']}")
        print(f"  Prediction: {result['prediction']}")

if __name__ == "__main__":
    main()

