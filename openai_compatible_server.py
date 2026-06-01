from __future__ import annotations

import json
import os
import base64
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nim_router import NimRouter, NimRouterError


Json = dict[str, Any]


def load_env_files() -> None:
    candidates = [
        Path("~/.config/nim-qwen-model-router/router.env").expanduser(),
        Path(".env"),
    ]
    explicit = os.environ.get("NIM_ROUTER_ENV_FILE")
    if explicit:
        candidates.insert(0, Path(explicit).expanduser())

    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#") or "=" not in item:
                continue
            if item.startswith("export "):
                item = item[len("export ") :].strip()
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value


load_env_files()

CONFIG_PATH = os.environ.get("NIM_ROUTER_CONFIG", "nim_router_config.json")
HOST = os.environ.get("NIM_ROUTER_HOST", "127.0.0.1")
PORT = int(os.environ.get("NIM_ROUTER_PORT", "8010"))
ROUTER_MODEL_PREFIX = os.environ.get("NIM_ROUTER_MODEL_PREFIX", "nvidia-router/")
GENERATED_ARTIFACT_DIR = Path(os.environ.get("NIM_ROUTER_ARTIFACT_DIR", "./artifacts")).expanduser()


def load_router() -> NimRouter:
    return NimRouter(CONFIG_PATH)


def extract_capability(model: str | None) -> str | None:
    if not model:
        return None
    if model.startswith(ROUTER_MODEL_PREFIX):
        return model[len(ROUTER_MODEL_PREFIX) :]
    return None


def has_image_content(messages: list[Json]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                    return True
    return False


def normalize_chat_response(router_response: Json, requested_model: str, routed_model: str | None = None) -> Json:
    artifact_message = save_artifacts_as_markdown(router_response)
    if artifact_message:
        return {
            "id": f"chatcmpl-router-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": artifact_message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
            "_router": router_response.get("_router", {"model": routed_model}),
        }

    if "choices" in router_response and isinstance(router_response["choices"], list):
        response = dict(router_response)
        response.setdefault("id", f"chatcmpl-router-{uuid.uuid4().hex}")
        response.setdefault("object", "chat.completion")
        response.setdefault("created", int(time.time()))
        response["model"] = requested_model
        response.setdefault("usage", {})
        response["_router"] = router_response.get("_router", {})
        for choice in response.get("choices", []):
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict) or message.get("content"):
                continue
            extracted = extract_tool_call_text(message)
            if extracted:
                message["content"] = extracted
        return response

    content = json.dumps(router_response, ensure_ascii=False)
    return {
        "id": f"chatcmpl-router-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
        "_router": router_response.get("_router", {"model": routed_model}),
    }


def save_artifacts_as_markdown(router_response: Json) -> str | None:
    artifacts = collect_artifacts(router_response)
    if not artifacts:
        return None

    GENERATED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        encoded = (
            artifact.get("base64")
            or artifact.get("b64_json")
            or artifact.get("data")
            or artifact.get("video")
            or artifact.get("image")
            or artifact.get("file")
        )
        if not isinstance(encoded, str) or not encoded:
            continue

        mime = str(artifact.get("mime_type") or artifact.get("mime") or "")
        encoded, mime = split_data_url(encoded, mime)
        try:
            raw = base64.b64decode(encoded)
        except Exception:
            continue

        artifact_type, ext = artifact_kind(raw, mime)
        filename = f"nim_{artifact_type}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{index}.{ext}"
        path = (GENERATED_ARTIFACT_DIR / filename).resolve()
        path.write_bytes(raw)
        if artifact_type == "image":
            links.append(f"Saved image: {path}\n\n![generated image]({path})")
        elif artifact_type == "video":
            links.append(f"Saved video: {path}\n\n[generated video]({path})")
        else:
            links.append(f"Saved artifact: {path}")

    if not links:
        return None
    return "\n\n".join(links)


def collect_artifacts(payload: Any) -> list[Json]:
    found: list[Json] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(collect_artifacts(item))
        return found
    if not isinstance(payload, dict):
        return found

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        found.extend(item for item in artifacts if isinstance(item, dict))

    for key in ("output", "outputs", "data", "result", "results", "response"):
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            found.extend(collect_artifacts(value))

    for key in ("base64", "b64_json", "video", "image", "file"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            found.append({key: value, "mime_type": payload.get("mime_type") or payload.get("mime") or ""})
    return found


def split_data_url(encoded: str, mime: str) -> tuple[str, str]:
    if not encoded.startswith("data:") or "," not in encoded:
        return encoded, mime
    header, body = encoded.split(",", 1)
    if ";" in header:
        mime = header[5:].split(";", 1)[0] or mime
    return body, mime


def extract_tool_call_text(message: Json) -> str | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    extracted: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or "tool_call"
        arguments = fn.get("arguments")
        if not isinstance(arguments, str) or not arguments:
            continue
        try:
            parsed = json.loads(arguments)
            rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            rendered = arguments
        extracted.append(f"{name}:\n```json\n{rendered}\n```")
    if not extracted:
        return None
    return "\n\n".join(extracted)


def image_extension(raw: bytes, mime: str = "") -> str:
    mime = mime.lower()
    if "png" in mime or raw.startswith(b"\x89PNG"):
        return "png"
    if "webp" in mime or raw.startswith(b"RIFF"):
        return "webp"
    return "jpg"


def artifact_kind(raw: bytes, mime: str = "") -> tuple[str, str]:
    mime = mime.lower()
    if "video" in mime or raw.startswith(b"\x00\x00\x00") or raw[4:8] == b"ftyp":
        return "video", "mp4"
    if "png" in mime or raw.startswith(b"\x89PNG"):
        return "image", "png"
    if "webp" in mime or raw.startswith(b"RIFF"):
        return "image", "webp"
    if "jpeg" in mime or "jpg" in mime or raw.startswith(b"\xff\xd8"):
        return "image", "jpg"
    return "artifact", "bin"


SAFE_CHAT_OVERRIDES = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
}

IMAGE_GENERATION_OVERRIDES = {
    "prompt",
    "height",
    "width",
    "steps",
    "samples",
    "seed",
    "cfg_scale",
    "mode",
    "image",
}

EMBEDDING_OVERRIDES = {
    "input",
    "input_type",
    "truncate",
    "dimensions",
    "encoding_format",
}

RERANK_OVERRIDES = {
    "query",
    "passages",
    "truncate",
}

VIDEO_GENERATION_OVERRIDES = {
    "image",
    "image_path",
    "seed",
    "cfg_scale",
    "motion_bucket_id",
}

OBJECT_DETECTION_OVERRIDES = {
    "input_video",
    "asset_id",
    "input_video_path",
    "video_path",
    "path",
    "threshold",
}

VISUAL_CHANGENET_OVERRIDES = {
    "reference_image",
    "test_image",
    "reference_asset_id",
    "test_asset_id",
    "reference_image_path",
    "before_image_path",
    "test_image_path",
    "after_image_path",
}

SPARSEDRIVE_OVERRIDES = {
    "scene_id",
}


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    server_version = "NimRouterOpenAICompatible/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json({"status": "ok"})
            return
        if self.path == "/v1/models":
            self.handle_models()
            return
        if self.path.startswith("/v1/models/"):
            self.handle_model()
            return
        if self.path in {"/v1/router/routes", "/router/routes"}:
            self.handle_routes()
            return
        if self.path in {"/v1/props", "/props"}:
            self.write_json({"context_length": 128000})
            return
        self.write_json({"error": {"message": "Not found", "type": "not_found"}}, status=404)

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions()
            return
        if self.path == "/v1/embeddings":
            self.handle_embeddings()
            return
        self.write_json({"error": {"message": "Not found", "type": "not_found"}}, status=404)

    def handle_models(self) -> None:
        try:
            router = load_router()
            data = [
                {
                    "id": f"{ROUTER_MODEL_PREFIX}{route.capability}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-nim-router",
                    "root": route.model,
                    "metadata": route.metadata,
                }
                for route in router.routes
            ]
            data.insert(
                0,
                {
                    "id": f"{ROUTER_MODEL_PREFIX}auto",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-nim-router",
                    "root": "auto",
                    "metadata": {"routing": "deterministic"},
                },
            )
            data.insert(
                1,
                {
                    "id": f"{ROUTER_MODEL_PREFIX}planner",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-nim-router",
                    "root": "adaptive-planner",
                    "metadata": {"routing": "local-qwen-planner"},
                },
            )
            data.insert(
                2,
                {
                    "id": f"{ROUTER_MODEL_PREFIX}fanout",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-nim-router",
                    "root": "bounded-fanout",
                    "metadata": {"routing": "parallel-fanout-aggregate"},
                },
            )
            self.write_json({"object": "list", "data": data})
        except NimRouterError as exc:
            self.write_error(str(exc), status=500)

    def handle_model(self) -> None:
        model_id = self.path[len("/v1/models/") :]
        try:
            router = load_router()
            for route in router.routes:
                virtual_id = f"{ROUTER_MODEL_PREFIX}{route.capability}"
                if model_id == virtual_id:
                    self.write_json(
                        {
                            "id": virtual_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local-nim-router",
                            "root": route.model,
                            "metadata": route.metadata,
                        }
                    )
                    return
            if model_id == f"{ROUTER_MODEL_PREFIX}auto":
                self.write_json(
                    {
                        "id": f"{ROUTER_MODEL_PREFIX}auto",
                        "object": "model",
                        "created": 0,
                        "owned_by": "local-nim-router",
                        "root": "auto",
                        "metadata": {"routing": "deterministic"},
                    }
                )
                return
            if model_id == f"{ROUTER_MODEL_PREFIX}planner":
                self.write_json(
                    {
                        "id": f"{ROUTER_MODEL_PREFIX}planner",
                        "object": "model",
                        "created": 0,
                        "owned_by": "local-nim-router",
                        "root": "adaptive-planner",
                        "metadata": {"routing": "local-qwen-planner"},
                    }
                )
                return
            if model_id == f"{ROUTER_MODEL_PREFIX}fanout":
                self.write_json(
                    {
                        "id": f"{ROUTER_MODEL_PREFIX}fanout",
                        "object": "model",
                        "created": 0,
                        "owned_by": "local-nim-router",
                        "root": "bounded-fanout",
                        "metadata": {"routing": "parallel-fanout-aggregate"},
                    }
                )
                return
            self.write_json({"error": {"message": "Model not found", "type": "not_found"}}, status=404)
        except NimRouterError as exc:
            self.write_error(str(exc), status=500)

    def handle_routes(self) -> None:
        try:
            router = load_router()
            self.write_json(
                {
                    "object": "router.routes",
                    "providers": router.provider_status(),
                    "routes": router.route_inventory(),
                    "virtual_models": [
                        f"{ROUTER_MODEL_PREFIX}auto",
                        f"{ROUTER_MODEL_PREFIX}planner",
                        f"{ROUTER_MODEL_PREFIX}fanout",
                    ],
                }
            )
        except NimRouterError as exc:
            self.write_error(str(exc), status=500)

    def handle_chat_completions(self) -> None:
        try:
            body = self.read_json()
            messages = body.get("messages")
            if not isinstance(messages, list):
                self.write_error("`messages` must be an array.", status=400)
                return

            requested_model = str(body.get("model") or f"{ROUTER_MODEL_PREFIX}auto")
            capability = extract_capability(requested_model)
            if capability == "auto":
                capability = None

            metadata = dict(body.get("metadata") or {})
            metadata["has_image"] = bool(metadata.get("has_image")) or has_image_content(messages)
            metadata["has_tools"] = bool(metadata.get("has_tools")) or bool(body.get("tools"))
            router = load_router()
            if capability in {"planner", "fanout"}:
                selected = router.select_route(messages, capability="general_chat", metadata=metadata)
            else:
                selected = router.select_route(messages, capability=capability, metadata=metadata)

            if selected.capability in {"image_generation", "image_edit"}:
                allowed_overrides = IMAGE_GENERATION_OVERRIDES
            elif selected.capability == "video_generation":
                allowed_overrides = VIDEO_GENERATION_OVERRIDES
            elif selected.capability == "object_detection":
                allowed_overrides = OBJECT_DETECTION_OVERRIDES
            elif selected.capability == "visual_changenet":
                allowed_overrides = VISUAL_CHANGENET_OVERRIDES
            elif selected.capability == "sparsedrive":
                allowed_overrides = SPARSEDRIVE_OVERRIDES
            elif selected.capability == "visual_retrieval_embedding":
                allowed_overrides = EMBEDDING_OVERRIDES
            elif selected.capability == "rerank":
                allowed_overrides = RERANK_OVERRIDES
            else:
                allowed_overrides = SAFE_CHAT_OVERRIDES
            payload_overrides = {key: value for key, value in body.items() if key in allowed_overrides}
            if "max_completion_tokens" in payload_overrides and "max_tokens" not in payload_overrides:
                payload_overrides["max_tokens"] = payload_overrides.pop("max_completion_tokens")

            result = router.invoke(
                messages,
                capability=capability,
                payload_overrides=payload_overrides,
                metadata=metadata,
            )
            response = normalize_chat_response(result, requested_model, routed_model=selected.model)
            if body.get("stream"):
                self.write_chat_completion_stream(response)
                return
            self.write_json(response)
        except NimRouterError as exc:
            self.write_error(str(exc), status=502)
        except json.JSONDecodeError:
            self.write_error("Invalid JSON request body.", status=400)

    def handle_embeddings(self) -> None:
        try:
            body = self.read_json()
            input_value = body.get("input")
            if input_value is None:
                self.write_error("`input` is required.", status=400)
                return

            requested_model = str(body.get("model") or f"{ROUTER_MODEL_PREFIX}qwen_embedding")
            capability = extract_capability(requested_model)
            if capability == "auto":
                capability = "qwen_embedding"
            if capability not in {None, "qwen_embedding", "visual_retrieval_embedding"}:
                capability = "qwen_embedding"

            allowed_overrides = {
                key: value
                for key, value in body.items()
                if key in EMBEDDING_OVERRIDES and key not in {"input"}
            }
            router = load_router()
            result = router.embed(input_value, model=capability, payload_overrides=allowed_overrides)
            response = dict(result)
            response.setdefault("object", "list")
            response["model"] = requested_model
            response["_router"] = result.get("_router", {})
            self.write_json(response)
        except NimRouterError as exc:
            self.write_error(str(exc), status=502)
        except json.JSONDecodeError:
            self.write_error("Invalid JSON request body.", status=400)

    def read_json(self) -> Json:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def write_error(self, message: str, *, status: int) -> None:
        self.write_json({"error": {"message": message, "type": "router_error"}}, status=status)

    def write_json(self, payload: Json, *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_chat_completion_stream(self, response: Json) -> None:
        choice = response.get("choices", [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        chunk_id = f"chatcmpl-router-stream-{uuid.uuid4().hex}"
        created = int(time.time())
        model = response.get("model") or f"{ROUTER_MODEL_PREFIX}auto"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        self.write_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        if content:
            self.write_sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )
        self.write_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}],
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def write_sse(self, payload: Json) -> None:
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), OpenAICompatibleHandler)
    print(f"NIM router OpenAI-compatible server listening on http://{HOST}:{PORT}")
    print(f"Use model '{ROUTER_MODEL_PREFIX}auto' or '{ROUTER_MODEL_PREFIX}<capability>'.")
    server.serve_forever()


if __name__ == "__main__":
    main()
