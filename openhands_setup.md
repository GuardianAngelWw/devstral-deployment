# Deploying Devstral-Small-2505 with OpenHands

This guide explains how to deploy Mistral AI's Devstral-Small-2505 model using vLLM server and OpenHands.

## Step 1: Launch vLLM Server

First, you need to launch a vLLM server with Devstral-Small-2505:

```bash
# Install vLLM if not already installed
pip install vllm

# Launch the vLLM server with Devstral-Small-2505
vllm serve mistralai/Devstral-Small-2505 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2
```

This will start an OpenAI-compatible server at `http://localhost:8000/v1`.

### Server Configuration Options

- `--tensor-parallel-size 2`: Adjust based on your GPU count
- `--port 8000`: Change if needed (default is 8000)
- `--host 0.0.0.0`: For external access (default is localhost only)

## Step 2: Launch OpenHands

The easiest way to use OpenHands is via Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands-state:/.openhands-state \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.38
```

This will start the OpenHands UI at `http://localhost:3000`.

## Step 3: Connect OpenHands to Your vLLM Server

1. Open `http://localhost:3000` in your browser
2. When prompted to connect to a server, use the advanced mode
3. Fill in the following fields:
   - **Custom Model**: `openai/mistralai/Devstral-Small-2505`
   - **Base URL**: `http://localhost:8000/v1` (or your server URL)
   - **API Key**: `token` (or any token you configured)

### Docker Network Configuration

If you're running both vLLM and OpenHands in Docker containers, use the following server URL for OpenHands to connect to the vLLM container:

```
http://host.docker.internal:8000/v1
```

## Step 4: Start Using Devstral in OpenHands

Once connected, you can start a new conversation and begin using Devstral-Small-2505 through the OpenHands interface.

## Troubleshooting

### Connection Issues
- Ensure vLLM server is running before starting OpenHands
- Check that the API URL is correctly formatted
- Verify network connectivity between OpenHands and vLLM

### Performance Issues
- Increase `tensor-parallel-size` for better performance on multi-GPU systems
- Adjust batch size with `--max-batch-size` in vLLM
- For better CPU utilization, try `--worker-use-ray`

### Memory Issues
- Reduce model loading memory with `--max-model-len` parameter
- Use `--gpu-memory-utilization 0.8` to control GPU memory usage

## Full Installation Script

Here's a complete script to set up both vLLM and OpenHands:

```bash
#!/bin/bash

# Install vLLM
pip install vllm

# Start vLLM server in background
nohup vllm serve mistralai/Devstral-Small-2505 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 > vllm.log 2>&1 &

echo "vLLM server started in background. Check vllm.log for progress."
echo "Once server is ready, launch OpenHands."

# Launch OpenHands
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands-state:/.openhands-state \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.38
```

Save this as `setup-openhands.sh`, make it executable with `chmod +x setup-openhands.sh`, and run it.