# Devstral-Small-2505 Deployment

This repository contains scripts to deploy [Mistral AI's Devstral-Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505) model as a local service or server.

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended) or Apple Silicon (32GB+ RAM)
- 100GB free storage space

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/GuardianAngelWw/devstral-deployment
cd devstral-deployment

# Install dependencies
pip install -r requirements.txt
```

## Usage Options

### Option 1: Simple CLI Chat

For a simple command-line chat interface:

```bash
python main.py
```

This will automatically download the model (if not already present) and launch a chat interface.

### Option 2: API Server with Flask

To deploy as a web API server:

```bash
python server.py --port 8000
```

This will start a server on port 8000 with the following endpoints:

- `POST /generate`: Generate text from a prompt
  ```json
  {
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 512,
    "temperature": 0.7
  }
  ```

- `POST /chat`: Chat with message history
  ```json
  {
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }
  ```

- `GET /health`: Health check endpoint

### Option 3: Transformers-based API Server

For a more standard Hugging Face Transformers deployment:

```bash
python inference.py --port 5000
```

## Docker Deployment

For containerized deployment, build and run the Docker image:

```bash
# Build the Docker image
docker build -t devstral-server .

# Run the container
docker run -p 8000:8000 devstral-server
```

## Using with OpenHands

To connect this server to OpenHands:

1. Start the server using one of the methods above
2. Configure your OpenHands client to point to your server endpoint
3. Ensure your OpenHands configuration includes the correct parameters for this model

## License

This deployment repository is MIT licensed, but please note that the Devstral-Small-2505 model itself is subject to Mistral AI's license terms.