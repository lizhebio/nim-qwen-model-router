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
- `GET /v1/router/routes` route/provider metadata discovery
- capability-based routing with virtual model names
- optional local-Qwen planner routing inspired by Router-R1
- optional bounded fanout and local-Qwen aggregation for hard text tasks
- local artifact saving for image/video results
- NVCF asset upload support for NVIDIA CV APIs
- environment-variable based secrets; no keys in config

## NVIDIA Links

- NVIDIA API catalog: <https://build.nvidia.com/>
- NVIDIA NIM API reference: <https://docs.api.nvidia.com/nim/reference>
- NVIDIA visual models APIs: <https://docs.api.nvidia.com/nim/reference/visual-models-apis>
- NVIDIA Cloud Functions asset API: <https://docs.api.nvidia.com/cloud-functions/reference/createasset>
- NVIDIA API key management: <https://build.nvidia.com/settings/api-keys>

## Prerequisites

You can run the router with only the local Qwen/BGE backend. NVIDIA credentials are required only for NVIDIA-backed routes such as image generation, vision chat, OCR, rerank, safety, and CV/video APIs.

### 1. NVIDIA Account and API Key

1. Sign in to the NVIDIA API catalog at <https://build.nvidia.com/>.
2. Open API key settings: <https://build.nvidia.com/settings/api-keys>.
3. Create an API key.
4. Put it in the persistent local config file before starting the router:

```bash
mkdir -p ~/.config/nim-qwen-model-router
chmod 700 ~/.config/nim-qwen-model-router
$EDITOR ~/.config/nim-qwen-model-router/router.env
```

Example:

```bash
NVIDIA_API_KEY="nvapi-..."
```

Do not commit this key. Keep it in `~/.config/nim-qwen-model-router/router.env`, your shell profile, secret manager, launch agent, or local `.env` file that is excluded by `.gitignore`.

Some NVIDIA hosted preview functions are account-gated. If a route returns an error such as:

```text
Function not found for account
HTTP 404 Not Found
```

the router is reaching NVIDIA, but the current NVIDIA account likely does not have access to that hosted function. This is especially common for preview CV/video endpoints.

### 2. NVIDIA Routes That Use Asset Uploads

Some NVIDIA CV APIs do not accept raw base64 media directly. They require uploading images/videos to NVIDIA Cloud Functions assets first, then passing asset IDs in request headers. This router includes helper logic for those routes:

- `nvidia-router/object_detection`
- `nvidia-router/visual_changenet`

Related NVIDIA docs:

- Asset creation/upload: <https://docs.api.nvidia.com/cloud-functions/reference/createasset>
- Retail object detection: <https://docs.api.nvidia.com/nim/reference/nvidia-retail-object-detection-infer>
- Visual ChangeNet: <https://docs.api.nvidia.com/nim/reference/nvidia-visual-changenet-infer>

### 3. Local Qwen/BGE Backend

For reliable default chat and embeddings, run or provide an OpenAI-compatible local gateway with:

- chat model: `Qwen3.5-397B-A17B-FP8`
- embedding model: `bge-large-zh-v1.5`

The gateway must support:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/embeddings
```

Then export:

```bash
export QWEN_BASE_URL="http://127.0.0.1:3000/v1"
export QWEN_API_KEY="..."
```

For persistent local use, put these values in:

```text
~/.config/nim-qwen-model-router/router.env
```

If your local gateway does not require an API key, leave `QWEN_API_KEY` unset.

### 4. Artifact Directory

Image and video results are saved locally before being returned as Markdown paths. Choose a writable directory:

```bash
export NIM_ROUTER_ARTIFACT_DIR="$PWD/artifacts"
```

For Hermes desktop/local workflows, you may prefer a user-visible directory:

```bash
export NIM_ROUTER_ARTIFACT_DIR="$HOME/Documents"
```

## Virtual Models

| Virtual model | Backend |
|---|---|
| `nvidia-router/auto` | automatic routing; normal chat falls back to local Qwen |
| `nvidia-router/planner` | local Qwen selects one validated capability before dispatch |
| `nvidia-router/fanout` | bounded text fanout plus local Qwen aggregation |
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

Clone the repository and create an environment file:

```bash
git clone git@github.com:lizhebio/nim-qwen-model-router.git
cd nim-qwen-model-router
cp .env.example .env
```

Set the values you need in the persistent local config:

```bash
mkdir -p ~/.config/nim-qwen-model-router
chmod 700 ~/.config/nim-qwen-model-router
$EDITOR ~/.config/nim-qwen-model-router/router.env
```

Example `router.env`:

```bash
NVIDIA_API_KEY="..."
QWEN_BASE_URL="http://127.0.0.1:3000/v1"
QWEN_API_KEY="..."
NIM_ROUTER_ARTIFACT_DIR="$PWD/artifacts"
```

Optional: validate the config before starting:

```bash
python3 tools/check_router.py
python3 -m py_compile nim_router.py openai_compatible_server.py
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

Provider secrets and local endpoints are controlled with environment variables. The server automatically loads these files at startup:

1. `NIM_ROUTER_ENV_FILE`, if set
2. `~/.config/nim-qwen-model-router/router.env`
3. `.env` in the repository root

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

Check Qwen tool-call pass-through through the running router:

```bash
python3 tools/check_router.py --live-tool-call
```

This sends an OpenAI-compatible `tools` + `tool_choice` request to `nvidia-router/qwen_chat` through `http://127.0.0.1:8010/v1` and fails if the response does not contain `tool_calls`.

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

Inspect route metadata and provider readiness:

```bash
curl http://127.0.0.1:8010/v1/router/routes
```

Test local-Qwen planner routing:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Authorization: Bearer local-router-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia-router/planner","messages":[{"role":"user","content":"生成一张极简风格的细胞实验流程图"}]}'
```

Test bounded fanout aggregation:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Authorization: Bearer local-router-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia-router/fanout","messages":[{"role":"user","content":"比较确定性路由和自适应路由的优缺点"}]}'
```

## Adaptive Routing

The Router-R1-inspired pieces are intentionally opt-in:

- `nvidia-router/auto` remains deterministic and is the recommended Hermes default.
- `nvidia-router/planner` asks local Qwen to choose a capability from the configured route inventory, then the gateway validates the selected capability before dispatching.
- `nvidia-router/fanout` calls a bounded allowlist of text routes, then asks local Qwen to synthesize the final answer.
- The planner cannot bypass config, image requirements, provider checks, or route allowlists.
- Route metadata in `nim_router_config.json` records modality, rough cost weight, latency class, artifact behavior, and account-gated status.

## Notes

- Generated images/videos are saved locally and returned as Markdown-friendly paths.
- Some NVIDIA hosted preview functions may return `404 Function not found` if the current NVIDIA account does not have access.
- Do not commit `.env`, generated artifacts, SQLite data, or API keys.
- Router-R1 was reviewed as a future adaptive-routing reference: [docs/router-r1-notes.md](docs/router-r1-notes.md).
