from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import os
import argparse

app = Flask(__name__)

MODEL_ID = "mistralai/Devstral-Small-2505"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print(f"Loading model {MODEL_ID} on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Model loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        'generated_text': response,
        'prompt': prompt
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Devstral API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)