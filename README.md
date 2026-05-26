# NIM + Qwen Model Router

OpenAI-compatible model router for agents. It exposes one local endpoint and routes requests to:

- NVIDIA NIM visual/multimodal APIs
- a local OpenAI-compatible Qwen chat backend
- a local OpenAI-compatible BGE embedding backend

The original use case is Hermes, but any OpenAI SDK-compatible agent can use it.

## Features

- `POST /v1/chat/completions` compatible facade
- `POST /v1/embeddings` compatible facade
- `GET /v1/models` model discovery
- capability-based routing with virtual model names
- local artifact saving for image/video results
- NVCF asset upload support for NVIDIA CV APIs
- environment-variable based secrets; no keys in config

## Virtual Models

| Virtual model | Backend |
|---|---|
| `nvidia-router/auto` | automatic routing; normal chat falls back to local Qwen |
| `nvidia-router/qwen_chat` | `Qwen3.5-397B-A17B-FP8` |
| `nvidia-router/qwen_embedding` | `bge-large-zh-v1.5` |
| `nvidia-router/general_chat` | `Qwen3.5-397B-A17B-FP8` |
| `nvidia-router/vision_chat` | `nvidia/nemotron-nano-12b-v2-vl` |
| `nvidia-router/document_parse` | `nvidia/nemotron-parse` |
| `nvidia-router/image_generation` | `black-forest-labs/flux.1-schnell` |
| `nvidia-router/image_edit` | `black-forest-labs/flux.1-kontext-dev` |
| `nvidia-router/video_generation` | `stabilityai/stable-video-diffusion` |
| `nvidia-router/object_detection` | `nvidia/retail-object-detection` |
| `nvidia-router/visual_changenet` | `nvidia/visual-changenet` |
| `nvidia-router/sparsedrive` | `nvidia/sparsedrive` |
| `nvidia-router/visual_retrieval_embedding` | `nvidia/llama-nemotron-embed-vl-1b-v2` |
| `nvidia-router/rerank` | `nvidia/llama-nemotron-rerank-vl-1b-v2` |
| `nvidia-router/safety` | `nvidia/llama-3.1-nemoguard-8b-content-safety` |

## Quick Start

Create an environment file:

```bash
cp .env.example .env
```

Set the values you need:

```bash
export NVIDIA_API_KEY="..."
export QWEN_BASE_URL="http://127.0.0.1:3000/v1"
export QWEN_API_KEY="..."
export NIM_ROUTER_ARTIFACT_DIR="$PWD/artifacts"
```

Start the local router:

```bash
python3 openai_compatible_server.py
```

The server listens at:

```text
http://127.0.0.1:8010/v1
```

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="local-router-key",
    base_url="http://127.0.0.1:8010/v1",
)

chat = client.chat.completions.create(
    model="nvidia-router/auto",
    messages=[{"role": "user", "content": "只回答 OK"}],
)
print(chat.choices[0].message.content)

embedding = client.embeddings.create(
    model="nvidia-router/qwen_embedding",
    input="测试 embedding 路由",
)
print(len(embedding.data[0].embedding))
```

## Hermes Configuration

Point Hermes at the router as its default provider:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8010/v1
  api_key: local-router-key
  default: nvidia-router/auto
```

You can keep your local Qwen provider in Hermes for direct fallback/debugging, but normal use should go through `nvidia-router/auto`.

## Configuration

Routes live in `nim_router_config.json`.

Provider secrets and local endpoints are controlled with environment variables:

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA hosted NIM APIs |
| `QWEN_BASE_URL` | local OpenAI-compatible Qwen/BGE gateway |
| `QWEN_API_KEY` | API key for the local Qwen/BGE gateway |
| `NIM_ROUTER_HOST` | server bind host, default `127.0.0.1` |
| `NIM_ROUTER_PORT` | server port, default `8010` |
| `NIM_ROUTER_ARTIFACT_DIR` | image/video artifact output directory |

## Validation

Check route configuration:

```bash
python3 tools/check_router.py
```

Compile Python files:

```bash
python3 -m py_compile nim_router.py openai_compatible_server.py
```

List models:

```bash
curl http://127.0.0.1:8010/v1/models
```

Test chat:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Authorization: Bearer local-router-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia-router/auto","messages":[{"role":"user","content":"只回答 OK"}]}'
```

Test embeddings:

```bash
curl http://127.0.0.1:8010/v1/embeddings \
  -H 'Authorization: Bearer local-router-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia-router/qwen_embedding","input":"hello"}'
```

## Notes

- Generated images/videos are saved locally and returned as Markdown-friendly paths.
- Some NVIDIA hosted preview functions may return `404 Function not found` if the current NVIDIA account does not have access.
- Do not commit `.env`, generated artifacts, SQLite data, or API keys.
