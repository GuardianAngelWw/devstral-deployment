#!/usr/bin/env python3
"""
Simple server script to deploy Devstral-Small-2505
Usage: python server.py --port 5000
"""

import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import torch
from flask import Flask, request, jsonify
import mistral_inference

# Set up paths and configurations
MODEL_ID = "mistralai/Devstral-Small-2505"
MODEL_DIR = Path.home() / "mistral_models" / "Devstral"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

def download_model():
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
        local_dir=MODEL_DIR
    )
    print(f"Model downloaded to {MODEL_DIR}")

def load_model():
    if not all((MODEL_DIR / f).exists() for f in ["params.json", "consolidated.safetensors"]):
        download_model()
    
    print("Loading Mistral model...")
    model = mistral_inference.load_model(str(MODEL_DIR))
    return model

# Load model at server startup
model = load_model()
print("Model loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from a prompt"""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    print(f"Generating with prompt: {prompt[:50]}...")
    response = model.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return jsonify({
        "generated_text": response,
        "prompt": prompt
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat API with message history support"""
    data = request.json
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing 'messages' in request"}), 400
    
    messages = data['messages']
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    # Convert to Mistral chat format
    response = model.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return jsonify({
        "response": response
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Devstral Model Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)