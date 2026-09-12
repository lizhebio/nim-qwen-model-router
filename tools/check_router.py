from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError


CHAT_TOOL_FIELDS = {"tools", "tool_choice", "parallel_tool_calls"}


def load_chat_override_fields(root: Path) -> set[str]:
    server_path = root / "openai_compatible_server.py"
    namespace: dict[str, object] = {}
    source = server_path.read_text(encoding="utf-8")
    start = source.index("SAFE_CHAT_OVERRIDES =")
    end = source.index("\n\nIMAGE_GENERATION_OVERRIDES", start)
    exec(source[start:end], namespace)
    overrides = namespace.get("SAFE_CHAT_OVERRIDES")
    if not isinstance(overrides, set):
        return set()
    return {str(item) for item in overrides}


def post_json(url: str, payload: dict, *, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Connection failed: {exc}") from exc
    except http.client.RemoteDisconnected as exc:
        raise RuntimeError(f"Router closed the connection without a response: {exc}") from exc


def post_sse(url: str, payload: dict, *, api_key: str, timeout: int) -> list[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    events: list[dict] = []
    try:
        with request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                events.append(json.loads(line[6:]))
    except (HTTPError, URLError, http.client.RemoteDisconnected) as exc:
        detail = exc.read().decode("utf-8", errors="replace") if isinstance(exc, HTTPError) else str(exc)
        raise RuntimeError(f"Streaming request failed: {detail}") from exc
    return events


def assert_tool_call_response(payload: dict) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"tool call response missing choices: {json.dumps(payload, ensure_ascii=False)[:1000]}")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"tool call response missing message: {json.dumps(payload, ensure_ascii=False)[:1000]}")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = message.get("content")
        raise RuntimeError(
            "tool call response did not include tool_calls. "
            f"finish_reason={choices[0].get('finish_reason')!r}, content={content!r}"
        )
    content = message.get("content")
    if isinstance(content, str) and "router_tool_call_probe" in content:
        raise RuntimeError("tool call was also rendered into message.content instead of staying tool_calls-only.")

    first = tool_calls[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"tool call item is not an object: {first!r}")
    fn = first.get("function")
    if not isinstance(fn, dict) or fn.get("name") != "router_tool_call_probe":
        raise RuntimeError(f"unexpected tool call function: {json.dumps(first, ensure_ascii=False)}")


def run_live_tool_call_check(base_url: str, api_key: str, timeout: int) -> None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "nvidia-router/auto",
        "messages": [
            {
                "role": "system",
                "content": "You are testing OpenAI-compatible tool calling. Call the requested tool exactly once.",
            },
            {
                "role": "user",
                "content": "Call router_tool_call_probe with city='Shanghai' and unit='celsius'. Do not answer directly.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "router_tool_call_probe",
                    "description": "Probe whether tool calling is passed through the model router.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city", "unit"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "router_tool_call_probe"}},
        "temperature": 0,
        "max_tokens": 128,
    }
    response = post_json(url, payload, api_key=api_key, timeout=timeout)
    assert_tool_call_response(response)
    router = response.get("_router", {})
    capability = router.get("capability")
    if capability != "tool_chat":
        raise RuntimeError(f"tool call probe routed to unexpected capability: {capability!r}")


def run_live_stream_tool_call_check(base_url: str, api_key: str, timeout: int) -> None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "nvidia-router/auto",
        "messages": [
            {"role": "system", "content": "Call the requested tool exactly once."},
            {"role": "user", "content": "Call router_tool_call_probe with city='Shanghai'."},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "router_tool_call_probe",
                "description": "Streaming probe.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "router_tool_call_probe"}},
        "stream": True,
        "temperature": 0,
        "max_tokens": 128,
    }
    events = post_sse(url, payload, api_key=api_key, timeout=timeout)
    deltas = []
    finish_reason = None
    rendered_content = ""
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            rendered_content += delta.get("content") or ""
            deltas.extend(delta.get("tool_calls") or [])
            finish_reason = choice.get("finish_reason") or finish_reason
    if not deltas:
        raise RuntimeError(f"stream contained no delta.tool_calls: {json.dumps(events, ensure_ascii=False)[:1500]}")
    if finish_reason != "tool_calls":
        raise RuntimeError(f"stream finish_reason was {finish_reason!r}, expected 'tool_calls'")
    if "router_tool_call_probe" in rendered_content:
        raise RuntimeError("stream rendered tool function name into delta.content")
    first = deltas[0]
    if (first.get("function") or {}).get("name") != "router_tool_call_probe":
        raise RuntimeError(f"unexpected streamed tool call: {json.dumps(first, ensure_ascii=False)}")


def run_live_image_generation_check(base_url: str, api_key: str, timeout: int) -> None:
    url = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": "nvidia-router/image_generation",
        "prompt": "a simple red apple on a plain white background",
        "size": "768x768",
        "n": 1,
        "response_format": "url",
        "steps": 5,
        "seed": 123,
    }
    response = post_json(url, payload, api_key=api_key, timeout=timeout)
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"image response missing data: {json.dumps(response, ensure_ascii=False)[:1200]}")
    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"image response item is not an object: {first!r}")
    path = first.get("path") or first.get("url")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"image response missing local path: {json.dumps(first, ensure_ascii=False)}")
    if not Path(path).is_file():
        raise RuntimeError(f"image path does not exist: {path}")
    if "![generated image](" not in str(first.get("markdown", "")):
        raise RuntimeError("image response is missing the Markdown image path")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NIM/Qwen router configuration.")
    parser.add_argument(
        "--live-tool-call",
        action="store_true",
        help="Call the running local router and verify tool-call routing avoids local Qwen.",
    )
    parser.add_argument(
        "--live-stream-tool-call",
        action="store_true",
        help="Call the running local router and verify streaming tool-call SSE deltas.",
    )
    parser.add_argument(
        "--live-image-generation",
        action="store_true",
        help="Call the running local router and verify standard image generation output and file persistence.",
    )
    parser.add_argument(
        "--router-url",
        default=os.environ.get("NIM_ROUTER_CHECK_URL", "http://127.0.0.1:8010/v1"),
        help="OpenAI-compatible router base URL for live checks.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NIM_ROUTER_CHECK_API_KEY", "local-router-key"),
        help="API key for live checks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("NIM_ROUTER_CHECK_TIMEOUT", "60")),
        help="Live check timeout in seconds.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / "nim_router_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    capabilities = [route["capability"] for route in config["routes"]]
    duplicates = sorted({item for item in capabilities if capabilities.count(item) > 1})
    if duplicates:
        print(f"Duplicate capabilities: {', '.join(duplicates)}", file=sys.stderr)
        return 1

    required = {"capability", "model", "method", "path", "match"}
    for route in config["routes"]:
        missing = required - set(route)
        if missing:
            print(f"{route.get('capability', '<unknown>')} missing: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
        metadata = route.get("metadata", {})
        if not isinstance(metadata, dict):
            print(f"{route['capability']} metadata must be an object", file=sys.stderr)
            return 1
        if metadata:
            if "modality" not in metadata:
                print(f"{route['capability']} metadata missing modality", file=sys.stderr)
                return 1
            if "cost_weight" in metadata and not isinstance(metadata["cost_weight"], (int, float)):
                print(f"{route['capability']} cost_weight must be numeric", file=sys.stderr)
                return 1

    qwen_tool_routes = [
        route
        for route in config["routes"]
        if route["capability"] in {"qwen_chat", "general_chat"}
        and route.get("provider") == "qwen-local"
        and route.get("openai_compatible")
    ]
    for route in qwen_tool_routes:
        metadata = route.get("metadata", {})
        if metadata.get("supports_tool_calls"):
            print(f"{route['capability']} should not be marked supports_tool_calls=true", file=sys.stderr)
            return 1

    tool_routes = [
        route
        for route in config["routes"]
        if route.get("metadata", {}).get("supports_tool_calls") and route.get("openai_compatible")
    ]
    if not tool_routes:
        print("No OpenAI-compatible tool-call route is configured", file=sys.stderr)
        return 1

    chat_overrides = load_chat_override_fields(root)
    missing_tool_fields = sorted(CHAT_TOOL_FIELDS - chat_overrides)
    if missing_tool_fields:
        print(f"SAFE_CHAT_OVERRIDES missing tool-call fields: {', '.join(missing_tool_fields)}", file=sys.stderr)
        return 1

    adaptive = config.get("adaptive", {})
    for capability in adaptive.get("fanout_capabilities", []):
        if capability not in capabilities:
            print(f"adaptive fanout capability not found: {capability}", file=sys.stderr)
            return 1

    print(f"OK: {len(capabilities)} routes")
    for capability in capabilities:
        print(f"- {capability}")
    print("OK: chat tool-call fields are passed through")

    if args.live_tool_call:
        try:
            run_live_tool_call_check(args.router_url, args.api_key, args.timeout)
        except RuntimeError as exc:
            print(f"Live tool-call check failed: {exc}", file=sys.stderr)
            return 1
        print("OK: live tool-call route probe")
    if args.live_stream_tool_call:
        try:
            run_live_stream_tool_call_check(args.router_url, args.api_key, args.timeout)
        except RuntimeError as exc:
            print(f"Live streaming tool-call check failed: {exc}", file=sys.stderr)
            return 1
        print("OK: live streaming tool-call route probe")
    if args.live_image_generation:
        try:
            run_live_image_generation_check(args.router_url, args.api_key, args.timeout)
        except RuntimeError as exc:
            print(f"Live image-generation check failed: {exc}", file=sys.stderr)
            return 1
        print("OK: live image-generation route probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
