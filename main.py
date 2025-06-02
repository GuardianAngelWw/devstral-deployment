#!/usr/bin/env python3
"""
CLI interface for Devstral-Small-2505
Usage: python main.py
"""

import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import torch
import time

# Set up paths and configurations
MODEL_ID = "mistralai/Devstral-Small-2505"
MODEL_DIR = Path.home() / "mistral_models" / "Devstral"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def download_model():
    """Download model files if not already present"""
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
        local_dir=MODEL_DIR
    )
    print(f"Model downloaded to {MODEL_DIR}")

def load_model():
    """Load the model using the correct inference backend"""
    try:
        import mistral_inference
        if not all((MODEL_DIR / f).exists() for f in ["params.json", "consolidated.safetensors"]):
            download_model()
        
        print("Loading Mistral model...")
        model = mistral_inference.load_model(str(MODEL_DIR))
        return model, "mistral"
    except ImportError:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("Loading model with transformers...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            return (model, tokenizer), "transformers"
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

def chat_loop(model, backend):
    """Interactive chat loop with the model"""
    print("\nDevstral-Small-2505 Chat")
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        start_time = time.time()
        
        if backend == "mistral":
            # Using native Mistral inference
            if not chat_history:
                # First message
                chat_history.append({"role": "user", "content": user_input})
                response = model.chat(chat_history)
                chat_history.append({"role": "assistant", "content": response})
            else:
                # Continuing conversation
                chat_history.append({"role": "user", "content": user_input})
                response = model.chat(chat_history)
                chat_history[-1] = {"role": "user", "content": user_input}
                chat_history.append({"role": "assistant", "content": response})
        else:
            # Using transformers
            model_obj, tokenizer = model
            if chat_history:
                formatted_prompt = ""
                for msg in chat_history:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        formatted_prompt += f"<|user|>\n{content}\n<|assistant|>\n"
                    else:
                        formatted_prompt += f"{content}\n"
                
                formatted_prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
            else:
                formatted_prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model_obj.device)
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.split("<|assistant|>\n")[-1].strip()
            
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
        
        elapsed_time = time.time() - start_time
        print(f"\nDevstral: {response}")
        print(f"\n[Generated in {elapsed_time:.2f}s]")

def main():
    parser = argparse.ArgumentParser(description="Devstral-Small-2505 CLI Chat Interface")
    args = parser.parse_args()
    
    model, backend = load_model()
    if model:
        chat_loop(model, backend)
    else:
        print("Failed to load model. Please check your installation.")

if __name__ == "__main__":
    main()