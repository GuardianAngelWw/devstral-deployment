import sys
from huggingface_hub import snapshot_download
from pathlib import Path
import subprocess
import os

MODEL_ID = "mistralai/Devstral-Small-2505"
MODEL_DIR = Path.home() / "mistral_models" / "Devstral"


def download_model():
    print("Downloading Devstral-Small-2505 model from Hugging Face...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
        local_dir=MODEL_DIR
    )
    print(f"Model downloaded to {MODEL_DIR}")


def run_chat():
    cmd = f"mistral-chat {MODEL_DIR} --instruct --max_tokens 300"
    print(f"Launching: {cmd}\n")
    subprocess.run(cmd, shell=True)


def main():
    if not all((MODEL_DIR / f).exists() for f in ["params.json", "consolidated.safetensors", "tekken.json"]):
        download_model()
    run_chat()

if __name__ == "__main__":
    main()
