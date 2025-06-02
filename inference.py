#!/usr/bin/env python3
"""
Transformers-based API server for Devstral-Small-2505
"""

import argparse
import os
import time
import torch
from flask import Flask, request, jsonify
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Devstral-Small-2505")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(Path.home() / "mistral_models" / "Devstral")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
model = None
tokenizer = None

def download_model():
    """Download model files if not already present"""
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["*.json", "*.bin", "*.safetensors"],
        local_dir=MODEL_DIR
    )
    print(f"Model downloaded to {MODEL_DIR}")

def load_model():
    """Load model using Transformers"""
    global model, tokenizer
    
    # Check if model files exist
    if not os.path.exists(MODEL_DIR / "config.json"):
        download_model()
    
    print("Loading model with transformers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    print("Model loaded successfully!")

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Generate completions (legacy endpoint)"""
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 128))
    temperature = float(data.get('temperature', 0.7))
    
    # Process with transformers
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Get response text
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    
    return jsonify({
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "text": output_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
            "total_tokens": len(outputs[0])
        },
        "system_fingerprint": "devstral-transformers-api"
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Chat completions API (OpenAI-compatible)"""
    data = request.json
    messages = data.get('messages', [])
    max_tokens = int(data.get('max_tokens', 128))
    temperature = float(data.get('temperature', 0.7))
    
    # Format messages into prompt
    prompt = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    
    # Process with transformers
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the newly generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant's response
    generated_text = full_output.split("<|assistant|>\n")[-1].strip()
    
    elapsed_time = time.time() - start_time
    
    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
            "total_tokens": len(outputs[0])
        },
        "system_fingerprint": "devstral-transformers-api"
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return available models (OpenAI compatibility)"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mistralai"
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers API Server for Devstral")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Load model at startup
    load_model()
    
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)