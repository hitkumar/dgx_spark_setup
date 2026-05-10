import torch
import vllm

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("capability:", torch.cuda.get_device_capability(0))

print("vllm:", vllm.__version__)

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-key",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "Explain what vLLM is in one paragraph."}
    ],
    temperature=0.2,
    max_tokens=200,
)

print(resp.choices[0].message.content)