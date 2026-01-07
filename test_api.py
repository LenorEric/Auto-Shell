#!/usr/bin/env python3
"""
Test OpenAI config (API key + base_url) and verify JSON response.

What this script does:
- Loads openai.api_key, base_url, model from config.yml
- Sends a minimal request
- Forces STRICT JSON output
- Asks the model to identify itself (model species / family)
"""

import json
import sys
from pathlib import Path

import yaml
from openai import OpenAI
from openai.types.shared_params import Reasoning


def load_config(path: str):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    openai_cfg = cfg.get("openai", {})

    api_key = openai_cfg.get("api_key")
    base_url = openai_cfg.get("base_url")
    model = openai_cfg.get("model")
    effort = openai_cfg.get("reasoning_effort")

    if not api_key:
        raise ValueError("api_key missing in config")
    if not model:
        raise ValueError("model missing in config")

    print(f"api_key: {api_key}")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"effort: {effort}")

    return api_key, base_url, model, effort

def eval_reasoning_effort(effort: str) -> Reasoning:
    effort_lower = effort.lower()
    if effort_lower == "none":
        return Reasoning(effort="none")
    elif effort_lower == "minimal":
        return Reasoning(effort="minimal")
    elif effort_lower == "low":
        return Reasoning(effort="low")
    elif effort_lower == "medium":
        return Reasoning(effort="medium")
    elif effort_lower == "high":
        return Reasoning(effort="high")
    elif effort_lower == "xhigh":
        return Reasoning(effort="xhigh")
    else:
        return Reasoning(effort="low")  # Default to low if unrecognized

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_openai_config.py config.yml")
        sys.exit(1)

    api_key, base_url, model, reasoning_effort = load_config(sys.argv[1])

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    # JSON schema for strict verification
    schema = {
        "name": "model_identity",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_name": {"type": "string"},
                "model_family": {"type": "string"},
                "provider": {"type": "string"},
                "api_ok": {"type": "boolean"},
                "miscellaneous": {"type": "string"},
            },
            "required": ["model_name", "model_family", "provider", "api_ok", "miscellaneous"],
        },
    }

    print("Sending test request...")

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a test endpoint. "
                    "Identify yourself accurately. "
                    "Output must strictly follow the provided JSON schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return your exact model name, model family/species, "
                    "and provider. Set api_ok=true."
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "strict": True,
                "schema": schema["schema"],
            }
        }
    )

    # Extract structured output
    try:
        raw = resp.output_text
        data = json.loads(raw)
    except Exception as e:
        print("❌ Failed to parse JSON response")
        print(e)
        print(resp)
        sys.exit(2)

    print("\n✅ API TEST SUCCESS")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
