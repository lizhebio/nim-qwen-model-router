from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


Json = dict[str, Any]


@dataclass(frozen=True)
class Route:
    capability: str
    description: str
    model: str
    method: str
    path: str
    provider: str
    openai_compatible: bool
    binary_response: bool
    priority: int
    match: Json
    metadata: Json


class NimRouterError(RuntimeError):
    pass


class NimRouter:
    def __init__(self, config_path: str | Path = "nim_router_config.json", api_key: str | None = None) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.base_url = config["base_url"].rstrip("/")
        self.providers = config.get("providers", {})
        self.timeout = int(config.get("default_timeout_seconds", 120))
        self.poll_interval = float(config.get("poll_interval_seconds", 2))
        self.max_poll_seconds = float(config.get("max_poll_seconds", 300))
        self.adaptive = config.get("adaptive", {})
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.routes = [
            Route(
                capability=item["capability"],
                description=item.get("description", ""),
                model=item["model"],
                method=item.get("method", "POST"),
                path=item["path"],
                provider=item.get("provider", "nvidia"),
                openai_compatible=bool(item.get("openai_compatible", False)),
                binary_response=bool(item.get("binary_response", False)),
                priority=int(item.get("priority", 0)),
                match=item.get("match", {}),
                metadata=item.get("metadata", {}),
            )
            for item in config["routes"]
        ]

    def select_route(
        self,
        messages: list[Json],
        *,
        images: list[str] | None = None,
        capability: str | None = None,
        metadata: Json | None = None,
    ) -> Route:
        if capability:
            for route in self.routes:
                if route.capability == capability:
                    return route
            raise NimRouterError(f"Unknown capability: {capability}")

        text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user").lower()
        has_image = bool(images) or bool(metadata and metadata.get("has_image"))

        scored: list[tuple[int, Route]] = []
        for route in self.routes:
            match = route.match
            if match.get("requires_image") and not has_image:
                continue

            score = route.priority
            matched_keywords = 0
            for keyword in match.get("keywords", []):
                if keyword.lower() in text:
                    matched_keywords += 1
                    score += 100

            is_image_default = bool(match.get("requires_image") and has_image)
            is_fallback = route.capability == "general_chat"
            if match.get("keywords") and not matched_keywords and not is_image_default and not is_fallback:
                continue
            scored.append((score, route))

        if not scored:
            return self._fallback_route()
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def invoke(
        self,
        messages: list[Json],
        *,
        images: list[str] | None = None,
        capability: str | None = None,
        payload_overrides: Json | None = None,
        metadata: Json | None = None,
    ) -> Json:
        if capability == "planner":
            return self.invoke_planned(
                messages,
                images=images,
                payload_overrides=payload_overrides,
                metadata=metadata,
            )
        if capability == "fanout":
            return self.invoke_fanout(messages, payload_overrides=payload_overrides, metadata=metadata)

        route = self.select_route(messages, images=images, capability=capability, metadata=metadata)
        payload = self._build_payload(route, messages, images=images, payload_overrides=payload_overrides)
        request_headers = payload.pop("_request_headers", None)
        response = self._request(
            route.method,
            route.path,
            payload,
            provider_name=route.provider,
            extra_headers=request_headers,
            accept="application/zip" if route.binary_response else "application/json",
            binary_response=route.binary_response,
        )

        if self._looks_async(response):
            return self._poll_until_done(response)
        response["_router"] = {
            "capability": route.capability,
            "model": route.model,
            "path": route.path,
            "metadata": route.metadata,
        }
        return response

    def invoke_planned(
        self,
        messages: list[Json],
        *,
        images: list[str] | None = None,
        payload_overrides: Json | None = None,
        metadata: Json | None = None,
    ) -> Json:
        decision = self.plan_route(messages, images=images, metadata=metadata)
        capability = str(decision.get("capability") or "general_chat")
        if capability in {"planner", "fanout", "qwen_embedding", "visual_retrieval_embedding"}:
            capability = "general_chat"

        try:
            route = self.select_route(messages, images=images, capability=capability, metadata=metadata)
        except NimRouterError:
            route = self._fallback_route()
            capability = route.capability

        if route.match.get("requires_image") and not (images or (metadata or {}).get("has_image")):
            route = self._fallback_route()
            capability = route.capability
            decision["fallback_reason"] = "Selected route requires image input."

        response = self.invoke(
            messages,
            images=images,
            capability=capability,
            payload_overrides=payload_overrides,
            metadata=metadata,
        )
        response.setdefault("_router", {})
        response["_router"]["planner"] = decision
        return response

    def plan_route(
        self,
        messages: list[Json],
        *,
        images: list[str] | None = None,
        metadata: Json | None = None,
    ) -> Json:
        heuristic_route = self.select_route(messages, images=images, metadata=metadata)
        planner_route = self._route_by_capability("general_chat")
        if not planner_route:
            return self._heuristic_decision(heuristic_route, reason="No local planner route is configured.")

        inventory = [
            {
                "capability": route.capability,
                "description": route.description,
                "modality": route.metadata.get("modality", []),
                "latency_class": route.metadata.get("latency_class", "unknown"),
                "account_gated": bool(route.metadata.get("account_gated", False)),
                "requires_image": bool(route.match.get("requires_image", False)),
            }
            for route in self.routes
            if route.capability not in {"qwen_embedding"}
        ]
        prompt = {
            "task": "Select exactly one router capability for the user request.",
            "rules": [
                "Return only JSON.",
                "Use capability=general_chat for normal conversation, writing, coding, or uncertain requests.",
                "Do not choose image/video/CV routes unless the user explicitly asks for that modality.",
                "Do not choose routes that require image input unless the request includes image input.",
            ],
            "has_image": bool(images) or bool(metadata and metadata.get("has_image")),
            "routes": inventory,
            "response_schema": {
                "capability": "one capability string from routes",
                "confidence": "number from 0 to 1",
                "reason": "short reason",
            },
        }
        planner_messages = [
            {
                "role": "system",
                "content": "You are a strict model-routing planner. Return compact JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "router_instruction": prompt,
                        "conversation": messages[-6:],
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            payload = {
                "model": planner_route.model,
                "messages": planner_messages,
                "temperature": 0,
                "max_tokens": 400,
            }
            response = self._request(
                planner_route.method,
                planner_route.path,
                payload,
                provider_name=planner_route.provider,
            )
            content = self._assistant_text(response)
            decision = self._parse_json_object(content)
        except Exception as exc:
            text = content if "content" in locals() else ""
            recovered = self._recover_capability_from_text(text)
            if recovered and self._has_capability(recovered):
                return {
                    "capability": recovered,
                    "confidence": 0.45,
                    "reason": f"Recovered capability from non-JSON planner output; parser error: {exc}",
                    "planner_model": planner_route.model,
                    "heuristic_capability": heuristic_route.capability,
                    "recovered": True,
                }
            return self._heuristic_decision(heuristic_route, reason=f"Planner failed; used heuristic: {exc}")

        capability = str(decision.get("capability") or "")
        if not self._has_capability(capability):
            return self._heuristic_decision(heuristic_route, reason=f"Planner selected unknown capability: {capability}")
        decision.setdefault("confidence", 0.5)
        decision.setdefault("reason", "Selected by local planner.")
        decision["planner_model"] = planner_route.model
        decision["heuristic_capability"] = heuristic_route.capability
        return decision

    def invoke_fanout(
        self,
        messages: list[Json],
        *,
        payload_overrides: Json | None = None,
        metadata: Json | None = None,
    ) -> Json:
        fanout_capabilities = self._fanout_capabilities(metadata)
        max_routes = int(self.adaptive.get("max_fanout_routes", 2))
        max_routes = max(1, min(max_routes, 4))
        selected_routes = [self.select_route([], capability=capability) for capability in fanout_capabilities[:max_routes]]
        safe_overrides = {
            key: value
            for key, value in (payload_overrides or {}).items()
            if key in {"temperature", "top_p", "max_tokens", "stop", "seed"}
        }

        results: list[Json] = []
        with ThreadPoolExecutor(max_workers=len(selected_routes)) as executor:
            futures = {
                executor.submit(self._invoke_route_for_fanout, route, messages, safe_overrides): route
                for route in selected_routes
            }
            for future in as_completed(futures):
                route = futures[future]
                try:
                    results.append({"route": route.capability, "model": route.model, "content": future.result()})
                except Exception as exc:
                    results.append({"route": route.capability, "model": route.model, "error": str(exc)})

        aggregator = self._route_by_capability("general_chat")
        if not aggregator:
            raise NimRouterError("fanout requires a general_chat route for aggregation.")

        aggregate_prompt = [
            {
                "role": "system",
                "content": (
                    "You synthesize routed model outputs into one concise final answer. "
                    "Ignore failed route outputs. Do not mention internal routing unless asked."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_conversation": messages,
                        "routed_outputs": results,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = {
            "model": aggregator.model,
            "messages": aggregate_prompt,
            "temperature": 0.2,
            "max_tokens": int(self.adaptive.get("max_fanout_aggregate_tokens", 1024)),
        }
        response = self._request(aggregator.method, aggregator.path, payload, provider_name=aggregator.provider)
        response["_router"] = {
            "capability": "fanout",
            "model": aggregator.model,
            "path": aggregator.path,
            "fanout": results,
            "metadata": {
                "max_routes": max_routes,
                "source": "router-r1-inspired bounded fanout",
            },
        }
        return response

    def _build_payload(
        self,
        route: Route,
        messages: list[Json],
        *,
        images: list[str] | None,
        payload_overrides: Json | None,
    ) -> Json:
        if route.capability == "image_generation":
            payload = {
                "prompt": self._last_user_text(messages),
                "height": 1024,
                "width": 1024,
                "steps": 4,
                "samples": 1,
                "seed": 0,
            }
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.capability == "image_edit":
            image = (images or [None])[0]
            payload = {
                "prompt": self._last_user_text(messages),
                "image": self._image_to_url(image) if image else None,
                "height": 1024,
                "width": 1024,
                "steps": 30,
                "samples": 1,
                "seed": 0,
            }
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.capability == "video_generation":
            image = self._coalesce_override(payload_overrides, "image", "image_path") or (images or [None])[0]
            if not image:
                raise NimRouterError("video_generation requires an image or image_path.")
            payload = {
                "image": self._image_to_url(str(image)),
                "seed": 0,
                "cfg_scale": 1.8,
                "motion_bucket_id": 127,
            }
            if payload_overrides:
                payload.update({k: v for k, v in payload_overrides.items() if k not in {"image_path"}})
            return payload

        if route.capability == "object_detection":
            asset_id = self._coalesce_override(payload_overrides, "input_video", "asset_id")
            if not asset_id:
                video_path = self._coalesce_override(payload_overrides, "input_video_path", "video_path", "path")
                if not video_path:
                    raise NimRouterError("object_detection requires input_video asset id or input_video_path.")
                asset_id = self._upload_asset(str(video_path), description="Retail object detection input video")
            payload = {
                "input_video": asset_id,
                "threshold": 0.9,
                "_request_headers": self._asset_headers(str(asset_id)),
            }
            if payload_overrides:
                payload.update(
                    {
                        k: v
                        for k, v in payload_overrides.items()
                        if k not in {"input_video_path", "video_path", "path", "asset_id"}
                    }
                )
            return payload

        if route.capability == "visual_changenet":
            reference_asset = self._coalesce_override(payload_overrides, "reference_image", "reference_asset_id")
            test_asset = self._coalesce_override(payload_overrides, "test_image", "test_asset_id")
            if not reference_asset:
                reference_path = self._coalesce_override(payload_overrides, "reference_image_path", "before_image_path")
                if not reference_path and images:
                    reference_path = images[0]
                if not reference_path:
                    raise NimRouterError("visual_changenet requires reference_image asset id or reference_image_path.")
                reference_asset = self._upload_asset(str(reference_path), description="Visual ChangeNet reference image")
            if not test_asset:
                test_path = self._coalesce_override(payload_overrides, "test_image_path", "after_image_path")
                if not test_path and images and len(images) > 1:
                    test_path = images[1]
                if not test_path:
                    raise NimRouterError("visual_changenet requires test_image asset id or test_image_path.")
                test_asset = self._upload_asset(str(test_path), description="Visual ChangeNet test image")
            asset_refs = f"{reference_asset},{test_asset}"
            payload = {
                "reference_image": reference_asset,
                "test_image": test_asset,
                "_request_headers": self._asset_headers(asset_refs),
            }
            if payload_overrides:
                payload.update(
                    {
                        k: v
                        for k, v in payload_overrides.items()
                        if k
                        not in {
                            "reference_asset_id",
                            "test_asset_id",
                            "reference_image_path",
                            "before_image_path",
                            "test_image_path",
                            "after_image_path",
                        }
                    }
                )
            return payload

        if route.capability == "sparsedrive":
            scene_id = self._coalesce_override(payload_overrides, "scene_id") or self._extract_scene_id(
                self._last_user_text(messages)
            )
            payload = {"scene_id": scene_id or "scene-0103"}
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.capability == "visual_retrieval_embedding":
            payload = {
                "model": route.model,
                "input": self._last_user_text(messages),
                "input_type": "query",
            }
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.capability == "qwen_embedding":
            payload = {
                "model": route.model,
                "input": self._last_user_text(messages),
            }
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.capability == "rerank":
            payload = {
                "model": route.model,
                "query": {"text": self._last_user_text(messages)},
                "passages": [{"text": self._last_user_text(messages)}],
                "truncate": "END",
            }
            if payload_overrides:
                payload.update(payload_overrides)
            return payload

        if route.openai_compatible:
            routed_messages = self._attach_images(messages, images or [])
            payload: Json = {
                "model": route.model,
                "messages": routed_messages,
            }
        else:
            payload = {
                "model": route.model,
                "messages": messages,
            }

        if payload_overrides:
            payload.update(payload_overrides)
        return payload

    def _attach_images(self, messages: list[Json], images: list[str]) -> list[Json]:
        if not images:
            return messages

        routed = [dict(message) for message in messages]
        user_index = next((i for i in range(len(routed) - 1, -1, -1) if routed[i].get("role") == "user"), None)
        if user_index is None:
            routed.append({"role": "user", "content": []})
            user_index = len(routed) - 1

        content = routed[user_index].get("content", "")
        parts: list[Json]
        if isinstance(content, list):
            parts = content
        else:
            parts = [{"type": "text", "text": str(content)}]

        for image in images:
            parts.append({"type": "image_url", "image_url": {"url": self._image_to_url(image)}})

        routed[user_index]["content"] = parts
        return routed

    def _last_user_text(self, messages: list[Json]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                        texts.append(str(part.get("text", "")))
                return "\n".join(text for text in texts if text).strip()
        return ""

    def _image_to_url(self, image: str) -> str:
        if image.startswith(("http://", "https://", "data:")):
            return image

        path = Path(image)
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{mime};base64,{encoded}"

    def _coalesce_override(self, payload_overrides: Json | None, *keys: str) -> Any:
        if not payload_overrides:
            return None
        for key in keys:
            value = payload_overrides.get(key)
            if value:
                return value
        return None

    def _extract_scene_id(self, text: str) -> str | None:
        for scene_id in ("scene-0103", "scene-0916", "scene-1073", "scene-0061"):
            if scene_id in text:
                return scene_id
        return None

    def _asset_headers(self, asset_refs: str) -> dict[str, str]:
        return {
            "NVCF-INPUT-ASSET-REFERENCES": asset_refs,
        }

    def _upload_asset(self, file_path: str, *, description: str) -> str:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise NimRouterError(f"Asset file does not exist: {path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        create_payload = {"contentType": content_type, "description": description}
        asset = self._request(
            "POST",
            "https://api.nvcf.nvidia.com/v2/nvcf/assets",
            create_payload,
            provider_name="nvidia",
            accept="application/json",
        )
        asset_id = asset.get("assetId") or asset.get("asset_id")
        upload_url = asset.get("uploadUrl") or asset.get("upload_url")
        if not asset_id or not upload_url:
            raise NimRouterError(f"NVCF asset creation returned an unexpected response: {asset}")

        self._upload_raw(
            str(upload_url),
            path.read_bytes(),
            headers={
                "Content-Type": content_type,
                "x-amz-meta-nvcf-asset-description": description,
            },
        )
        time.sleep(1)
        return str(asset_id)

    def _upload_raw(self, url: str, data: bytes, *, headers: dict[str, str]) -> None:
        req = request.Request(url, data=data, method="PUT", headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise NimRouterError(f"NVCF asset upload failed: HTTP {exc.code}: {detail}") from exc

    def embed(self, input_value: Any, *, model: str | None = None, payload_overrides: Json | None = None) -> Json:
        capability = model if model and self._has_capability(model) else "qwen_embedding"
        route = self.select_route([], capability=capability)
        payload: Json = {
            "model": route.model,
            "input": input_value,
        }
        if payload_overrides:
            payload.update(payload_overrides)
        response = self._request(route.method, route.path, payload, provider_name=route.provider)
        response["_router"] = {
            "capability": route.capability,
            "model": route.model,
            "path": route.path,
            "metadata": route.metadata,
        }
        return response

    def _has_capability(self, capability: str) -> bool:
        return any(route.capability == capability for route in self.routes)

    def _route_by_capability(self, capability: str) -> Route | None:
        for route in self.routes:
            if route.capability == capability:
                return route
        return None

    def _invoke_route_for_fanout(self, route: Route, messages: list[Json], payload_overrides: Json) -> str:
        payload = self._build_payload(route, messages, images=None, payload_overrides=payload_overrides)
        response = self._request(route.method, route.path, payload, provider_name=route.provider)
        return self._assistant_text(response) or json.dumps(response, ensure_ascii=False)

    def _fanout_capabilities(self, metadata: Json | None) -> list[str]:
        configured = (metadata or {}).get("fanout_capabilities") or self.adaptive.get("fanout_capabilities")
        if not configured:
            configured = ["qwen_chat", "general_chat"]
        allowed: list[str] = []
        seen_models: set[tuple[str, str]] = set()
        for capability in configured:
            if not isinstance(capability, str) or capability in {"planner", "fanout"}:
                continue
            route = self._route_by_capability(capability)
            if not route or route.match.get("requires_image") or not route.openai_compatible:
                continue
            if route.capability in {"qwen_embedding", "visual_retrieval_embedding"}:
                continue
            identity = (route.provider, route.model)
            if identity in seen_models and allowed:
                continue
            seen_models.add(identity)
            allowed.append(route.capability)
        return allowed or ["general_chat"]

    def _assistant_text(self, response: Json) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _parse_json_object(self, text: str) -> Json:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise NimRouterError("Planner returned non-object JSON.")
        return parsed

    def _recover_capability_from_text(self, text: str) -> str | None:
        lower = text.lower()
        capabilities = sorted((route.capability for route in self.routes), key=len, reverse=True)
        for capability in capabilities:
            if capability.lower() in lower:
                return capability
        for capability in capabilities:
            dashed = capability.replace("_", "-").lower()
            spaced = capability.replace("_", " ").lower()
            if dashed in lower or spaced in lower:
                return capability
        return None

    def _heuristic_decision(self, route: Route, *, reason: str) -> Json:
        return {
            "capability": route.capability,
            "confidence": 0.4,
            "reason": reason,
            "heuristic": True,
        }

    def route_inventory(self) -> list[Json]:
        return [
            {
                "capability": route.capability,
                "description": route.description,
                "model": route.model,
                "provider": route.provider,
                "openai_compatible": route.openai_compatible,
                "priority": route.priority,
                "match": route.match,
                "metadata": route.metadata,
            }
            for route in self.routes
        ]

    def provider_status(self) -> Json:
        status: Json = {}
        for name, provider in self.providers.items():
            resolved = self._resolved_provider(provider)
            api_key_env = provider.get("api_key_env")
            base_url_env = provider.get("base_url_env")
            status[name] = {
                "base_url": resolved.get("base_url"),
                "api_key_configured": bool(api_key_env and os.environ.get(str(api_key_env))),
                "api_key_env": api_key_env,
                "base_url_env": base_url_env,
                "base_url_overridden": bool(base_url_env and os.environ.get(str(base_url_env))),
            }
        return status

    def _request(
        self,
        method: str,
        path: str,
        payload: Json | None = None,
        *,
        provider_name: str | None = None,
        extra_headers: dict[str, str] | None = None,
        accept: str = "application/json",
        binary_response: bool = False,
    ) -> Json:
        route_provider = self._provider_for_path(path, provider_name=provider_name)
        base_url = str(route_provider.get("base_url") or self.base_url).rstrip("/")
        url = path if path.startswith("http") else f"{base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        api_key = self._api_key_for_provider(route_provider)
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        req = request.Request(
            url,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
                status_header = response.headers.get("NVCF-STATUS")
                request_id = response.headers.get("NVCF-REQID")
                if response.status == 202 or request_id:
                    return {
                        "status": (status_header or "pending").lower(),
                        "requestId": request_id,
                    }
                if binary_response:
                    content_type = response.headers.get("Content-Type", "")
                    return {
                        "artifacts": [
                            {
                                "base64": base64.b64encode(raw).decode("ascii"),
                                "mime_type": content_type or accept,
                            }
                        ]
                    }
                data = raw.decode("utf-8")
                return json.loads(data) if data else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise NimRouterError(f"NVIDIA NIM request failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise NimRouterError(f"Provider request failed: {exc}") from exc

    def _provider_for_path(self, path: str, *, provider_name: str | None = None) -> Json:
        if provider_name and provider_name in self.providers:
            return self._resolved_provider(self.providers[provider_name])
        for provider in self.providers.values():
            base_url = str(provider.get("base_url", "")).rstrip("/")
            if base_url and path.startswith(base_url):
                return self._resolved_provider(provider)
        return self._resolved_provider(self.providers.get("nvidia", {"base_url": self.base_url, "api_key_env": "NVIDIA_API_KEY"}))

    def _resolved_provider(self, provider: Json) -> Json:
        resolved = dict(provider)
        base_url_env = resolved.get("base_url_env")
        if base_url_env and os.environ.get(str(base_url_env)):
            resolved["base_url"] = os.environ[str(base_url_env)]
        return resolved

    def _api_key_for_provider(self, provider: Json) -> str | None:
        env_name = provider.get("api_key_env")
        if env_name and os.environ.get(str(env_name)):
            return os.environ[str(env_name)]
        if provider.get("api_key"):
            return str(provider["api_key"])
        if provider.get("name") == "nvidia":
            if not self.api_key:
                raise NimRouterError("Set NVIDIA_API_KEY to use NVIDIA NIM routes.")
            return self.api_key
        return None

    def _looks_async(self, response: Json) -> bool:
        return any(key in response for key in ("status_url", "poll_url", "request_id", "requestId", "id")) and str(
            response.get("status", "")
        ).lower() in {"pending", "running", "submitted", "processing", "queued", "fulfilled"}

    def _poll_until_done(self, response: Json) -> Json:
        poll_url = response.get("status_url") or response.get("poll_url")
        request_id = response.get("request_id") or response.get("requestId")
        if not poll_url and request_id:
            poll_url = f"https://ai.api.nvidia.com/v1/status/{request_id}"
        if not poll_url:
            response["_router_warning"] = "Response looks async, but no poll URL was returned."
            return response

        started = time.monotonic()
        while time.monotonic() - started < self.max_poll_seconds:
            time.sleep(self.poll_interval)
            current = self._request("GET", poll_url)
            status = str(current.get("status", "")).lower()
            if status in {"succeeded", "success", "completed", "done", "fulfilled"}:
                return current
            if status in {"failed", "error", "cancelled"}:
                raise NimRouterError(f"NVIDIA NIM async job failed: {json.dumps(current, ensure_ascii=False)}")

        raise NimRouterError(f"NVIDIA NIM async job timed out after {self.max_poll_seconds} seconds.")

    def _fallback_route(self) -> Route:
        for route in self.routes:
            if route.capability == "general_chat":
                return route
        return sorted(self.routes, key=lambda route: route.priority, reverse=True)[0]


def route_for_agent_tool(args: Json) -> Json:
    router = NimRouter(args.get("config_path", "nim_router_config.json"), api_key=args.get("api_key"))
    result = router.invoke(
        args["messages"],
        images=args.get("images"),
        capability=args.get("capability"),
        payload_overrides=args.get("payload_overrides"),
        metadata=args.get("metadata"),
    )
    return result


if __name__ == "__main__":
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful multimodal assistant."},
            {"role": "user", "content": "请描述这张截图里的关键信息。"},
        ],
        "images": ["./example.png"],
    }
    print(json.dumps({"selected": NimRouter().select_route(**sample).__dict__}, ensure_ascii=False, indent=2))
