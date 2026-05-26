# Router-R1 Research Notes

Reference project: <https://github.com/ulab-uiuc/Router-R1>

Router-R1 is a research implementation of a learned, multi-turn LLM router. It is useful as a design reference, but it is not a drop-in replacement for this project's lightweight OpenAI-compatible gateway.

## What Router-R1 Does

Router-R1 trains a base reasoning model to decide when it needs help from specialist LLMs. During generation, the base model can emit a structured routing request:

```text
<search> LLM-Name:Delegated question </search>
```

The route service calls one or more selected specialist models, injects their returned information back into the prompt, and lets the base model continue reasoning until it emits a final answer:

```text
<answer> Final response </answer>
```

At a high level:

```text
User query
  -> base router/reasoner model
  -> optional <search> Model:Question </search>
  -> route service calls specialist model(s)
  -> <information>...</information> is appended
  -> base model aggregates
  -> final answer
```

## Relevant Components

- `data_process/prompt_pool.py`: prompt templates that teach the base model how to request specialist models with `<search>` tags and how to produce final `<answer>` tags.
- `router_r1/llm_agent/route_service.py`: OpenAI SDK based service wrapper that dispatches delegated queries to multiple backend models, including a small token-cost estimate table.
- `infer_vllm.py`: vLLM inference loop that extracts `<search>` requests, calls the routing pool, appends `<information>`, and repeats for a bounded number of turns.
- `train.sh` and `test.sh`: veRL PPO training and evaluation entry points. This is the heavy research stack that makes Router-R1 a learned router rather than a simple rules engine.

## Difference From This Router

This repository currently implements an OpenAI-compatible production gateway:

- Hermes or any OpenAI SDK client talks to `http://127.0.0.1:8010/v1`.
- Virtual model names such as `nvidia-router/auto` select local Qwen, local BGE, or NVIDIA NIM routes.
- Routing is deterministic and capability based.
- NVIDIA-specific responses can be normalized into Markdown-friendly local artifact paths.
- Local Qwen/BGE routes continue to work even when NVIDIA credentials or preview function access are unavailable.

Router-R1 operates one layer above that:

- It is a learned reasoning policy for choosing specialist models.
- It requires a base model, prompt protocol, routing loop, and optionally RL training.
- It is primarily designed for text-specialist routing and answer aggregation.
- It is not directly shaped as an OpenAI-compatible HTTP gateway for agents like Hermes.

## What We Should Borrow

The most useful Router-R1 ideas for this project are practical and incremental:

1. Route metadata

   Add richer metadata to `nim_router_config.json`, such as rough cost, latency class, modality, account-gated status, and health-check behavior. This gives the router enough context to make better choices without requiring RL.

2. Explicit planner route

   Add an optional route-planning mode powered by the local Qwen backend. For ambiguous requests, Qwen can select a capability from the known route inventory and return a small structured decision object. The actual gateway should still enforce allowlists and route availability.

3. Parallel fanout for hard text tasks

   For complex text-only questions, optionally call multiple specialist routes in parallel and aggregate the result. This mirrors Router-R1's routing pool, but can be implemented with normal Python concurrency and OpenAI-compatible clients.

4. Budget controls

   Router-R1's bounded multi-turn loop is important. Any planner/fanout mode here should have explicit limits: max routed calls, max latency, max tokens, and allowed modalities.

5. Cost and token accounting

   Track approximate input/output token usage per route. This is especially useful when choosing between local Qwen, NVIDIA hosted models, and future remote providers.

## What We Should Not Copy Directly

- The full veRL PPO training stack is too heavy for this repo's immediate goal.
- vLLM-specific inference code should not become a dependency of the gateway.
- Visible `<think>`-style reasoning should not be exposed through Hermes.
- Hardcoded Router-R1 model names should not replace this repo's config-driven route inventory.
- A model-generated routing decision should never bypass local validation, provider health checks, or capability allowlists.

## Recommended Integration Roadmap

### Phase 1: Config Metadata

Extend each route with optional metadata:

```json
{
  "capability": "image_generation",
  "modality": ["text", "image"],
  "latency_class": "remote",
  "cost_weight": 1.0,
  "account_gated": true,
  "health_required": true
}
```

The first implementation can remain deterministic, using the metadata only for discovery, diagnostics, and future planning.

### Phase 2: Planner Mode

Add a virtual model such as:

```text
nvidia-router/planner
```

It would call local Qwen with a compact inventory of available routes and request JSON like:

```json
{
  "capability": "image_generation",
  "confidence": 0.86,
  "reason": "The user asks to generate an image.",
  "requires_artifact": true
}
```

The gateway should validate the selected capability against config before invoking anything.

### Phase 3: Fanout and Aggregate

For advanced text tasks, add a bounded fanout mode:

```text
nvidia-router/fanout
```

This can query two or more approved text-capable backends in parallel and ask local Qwen to synthesize the final response. This should be opt-in because it increases latency and cost.

### Phase 4: Learning From Logs

Only consider Router-R1-style RL after enough real usage logs exist:

- user request
- selected route
- success or error
- latency
- cost estimate
- user correction or retry signal

Before RL, simpler supervised route classification or preference scoring will likely be more maintainable.

## Practical Recommendation

For this repository, Router-R1 should be treated as a routing-policy reference rather than a dependency. The best near-term move is:

1. Keep the current OpenAI-compatible facade stable for Hermes.
2. Add route metadata and health reporting.
3. Add an optional local-Qwen planner behind a virtual model.
4. Keep deterministic fallback to `Qwen3.5-397B-A17B-FP8` for normal chat.
5. Avoid routing ordinary text prompts to NVIDIA preview endpoints by default.

This keeps the router reliable today while leaving a clean path toward Router-R1-style adaptive routing later.

## Implemented In This Router

The following Router-R1-inspired pieces have been merged into this repository:

- Route metadata in `nim_router_config.json`, including modality, latency class, rough cost weight, account-gated status, artifact behavior, and NVCF asset requirements.
- `GET /v1/router/routes`, which exposes route metadata and provider readiness without exposing API keys.
- `nvidia-router/planner`, an opt-in virtual model that asks local Qwen to choose a capability, then validates that capability before dispatch.
- `nvidia-router/fanout`, an opt-in bounded fanout/aggregation path for text routes.
- Basic budget controls in config: `max_fanout_routes` and `max_fanout_aggregate_tokens`.
- Safer defaults: `nvidia-router/auto` remains deterministic and continues to fall back to local Qwen for normal chat.

The heavier Router-R1 pieces, especially veRL PPO training and vLLM-specific inference, are intentionally not included.
