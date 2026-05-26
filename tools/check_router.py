from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
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

    adaptive = config.get("adaptive", {})
    for capability in adaptive.get("fanout_capabilities", []):
        if capability not in capabilities:
            print(f"adaptive fanout capability not found: {capability}", file=sys.stderr)
            return 1

    print(f"OK: {len(capabilities)} routes")
    for capability in capabilities:
        print(f"- {capability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
