# dgx_spark_setup

## Setup
Create pyproject.toml, run uv sync.

To check if huggingface login is working.
uv run hf auth whoami

Activate venv cmd
source /home/htkumar/dgx_spark_setup/.venv/bin/activate

## vllm test

Start the server
vllm serve Qwen/Qwen3-0.6B   --host 127.0.0.1   --port 8000   --max-model-len 4096   --gpu-memory-utilization 0.80

Run this cmd to test

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Write a tiny Python function to reverse a string. Return only the code."}
    ],
    "temperature": 0.2,
    "max_tokens": 200,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'