#!/usr/bin/env python3
"""
Flask API server for Devstral-Small-2505 using native Mistral inference
"""

import argparse
import json
import os
import time
from pathlib import Path
from flask import Flask, request, jsonify
from huggingface_hub import snapshot_download

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Devstral-Small-2505")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(Path.home() / "mistral_models" / "Devstral")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
model = None

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
    """Load model using Mistral inference library"""
    global model
    
    # Check if model files exist
    if not all((MODEL_DIR / f).exists() for f in ["params.json", "consolidated.safetensors"]):
        download_model()
    
    try:
        import mistral_inference
        print("Loading model with Mistral inference...")
        model = mistral_inference.load_model(str(MODEL_DIR))
        print("Model loaded successfully!")
        return True
    except ImportError:
        print("Mistral inference library not found.")
        print("Please install it with: pip install mistral-inference")
        return False

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Chat completions API endpoint (OpenAI-compatible)"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    messages = data.get('messages', [])
    max_tokens = int(data.get('max_tokens', 512))
    temperature = float(data.get('temperature', 0.7))
    
    # Convert to Mistral inference format
    mistral_messages = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role in ["user", "assistant", "system"]:
            mistral_messages.append({"role": role, "content": content})
    
    start_time = time.time()
    
    # Generate response
    response = model.chat(
        mistral_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    elapsed_time = time.time() - start_time
    
    # Calculate token counts (approximate)
    prompt_tokens = sum(len(msg.get('content', '').split()) * 1.3 for msg in messages)
    completion_tokens = len(response.split()) * 1.3
    
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
                    "content": response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens)
        },
        "system_fingerprint": "devstral-native-api"
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Text completions API endpoint (OpenAI-compatible)"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 512))
    temperature = float(data.get('temperature', 0.7))
    
    # Create a single message with the prompt
    messages = [{"role": "user", "content": prompt}]
    
    start_time = time.time()
    
    # Generate response
    response = model.chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    elapsed_time = time.time() - start_time
    
    # Calculate token counts (approximate)
    prompt_tokens = len(prompt.split()) * 1.3
    completion_tokens = len(response.split()) * 1.3
    
    return jsonify({
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "text": response,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens)
        },
        "system_fingerprint": "devstral-native-api"
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatibility)"""
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
    if model is None:
        return jsonify({"status": "not ready", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral Inference API Server for Devstral")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Load model at startup
    if load_model():
        print(f"Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("Failed to load model. Server not started.")