#!/usr/bin/env python3
"""
Requires:
  pip install openai pyyaml
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------

@dataclass
class AppConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    reasoning_effort: str
    timeout_seconds: int
    max_output_chars: int
    workdir: Optional[str]
    env: Dict[str, str]
    denylist_regex: List[str]
    allowlist_prefixes: List[str]
    require_allowlist: bool
    show_raw_model_json: bool


def load_config(path: str) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    openai_cfg = data.get("openai", {})
    agent_cfg = data.get("agent", {})
    safety_cfg = data.get("safety", {})

    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key (set openai.api_key or OPENAI_API_KEY).")

    return AppConfig(
        api_key=api_key,
        base_url=openai_cfg.get("base_url"),
        model=openai_cfg.get("model", "gpt-5-chat-latest"),
        reasoning_effort=openai_cfg.get("reasoning_effort", "low"),
        timeout_seconds=int(agent_cfg.get("timeout_seconds", 30)),
        max_output_chars=int(agent_cfg.get("max_output_chars", 12000)),
        workdir=agent_cfg.get("workdir"),
        env={str(k): str(v) for k, v in (agent_cfg.get("env", {}) or {}).items()},
        denylist_regex=list(safety_cfg.get("denylist_regex", []) or []),
        allowlist_prefixes=list(safety_cfg.get("allowlist_prefixes", []) or []),
        require_allowlist=bool(safety_cfg.get("require_allowlist", False)),
        show_raw_model_json=bool(agent_cfg.get("show_raw_model_json", False)),
    )


# -----------------------------
# Shell execution
# -----------------------------

def detect_shell() -> Tuple[str, List[str]]:
    sysname = platform.system().lower()
    if "windows" in sysname:
        # PowerShell
        return ("powershell", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"])
    # bash (Linux/macOS)
    return ("bash", ["bash", "-lc"])


def run_command(
    shell_name: str,
    shell_prefix: List[str],
    command: str,
    timeout_seconds: int,
    max_output_chars: int,
    workdir: Optional[str],
    extra_env: Dict[str, str],
) -> Dict[str, Any]:
    cwd = workdir or os.getcwd()
    env = os.environ.copy()
    env.update(extra_env)

    start = time.time()
    try:
        proc = subprocess.run(
            shell_prefix + [command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start
        stdout = (proc.stdout or "")[:max_output_chars]
        stderr = (proc.stderr or "")[:max_output_chars]
        return {
            "shell": shell_name,
            "command": command,
            "cwd": cwd,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            "elapsed_seconds": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        stdout = (e.stdout or "")[:max_output_chars] if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "")[:max_output_chars] if isinstance(e.stderr, str) else ""
        return {
            "shell": shell_name,
            "command": command,
            "cwd": cwd,
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "elapsed_seconds": round(elapsed, 3),
        }


# -----------------------------
# Safety checks
# -----------------------------

def is_denied(command: str, denylist_regex: List[str]) -> Optional[str]:
    for pattern in denylist_regex:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return pattern
    return None


def is_allowed_by_prefix(command: str, allowlist_prefixes: List[str]) -> bool:
    if not allowlist_prefixes:
        return True
    stripped = command.lstrip()
    return any(stripped.startswith(pfx) for pfx in allowlist_prefixes)


# -----------------------------
# Model I/O (Structured JSON)
# -----------------------------

def build_json_schema() -> Dict[str, Any]:
    # One-command-per-step contract.
    return {
        "name": "smart_shell_step",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "done": {"type": "boolean"},
                "command": {"type": ["string", "null"], "description": "Exactly one shell command to run, or null if done=true."},
                "shell": {"type": "string", "enum": ["bash", "powershell"], "description": "Which shell the command targets."},
                "explanation": {"type": "string", "description": "Short explanation of why this command is next."},
                "risk": {"type": "string", "enum": ["low", "medium", "high"], "description": "Risk level of this command."},
                "confirmation_prompt": {"type": "string", "description": "Prompt shown to the user before execution."},
                "expected_outcome": {"type": "string", "description": "What success looks like for this step."},
                "done_summary": {"type": ["string", "null"], "description": "If done=true, summarize what was accomplished."},
                "next_goal_prompt": {"type": ["string", "null"], "description": "If done=true, ask user what to do next."},
            },
            "required": ["done", "command", "shell", "explanation", "risk", "confirmation_prompt", "expected_outcome", "done_summary", "next_goal_prompt"],
        },
    }


def extract_output_text(resp: Any) -> str:
    # Responses API returns content in output items; easiest is to use SDK convenience if available.
    # Fallback: stringify.
    try:
        return resp.output_text  # type: ignore[attr-defined]
    except Exception:
        return json.dumps(resp, default=str)


def parse_step_json(raw_text: str) -> Dict[str, Any]:
    # With structured outputs, raw_text should be JSON.
    return json.loads(raw_text)


def build_system_instructions(shell_name: str) -> str:
    return f"""
You are SmartShell, an assistant that completes a user's goal by proposing ONE terminal command at a time.

Hard rules:
- Output MUST be valid JSON matching the provided schema (no markdown, no extra keys).
- Propose EXACTLY ONE command per step when done=false.
- The command MUST target the user's shell: "{shell_name}" only.
- Prefer safe, read-only inspection commands first.
- If a step is risky (deletes data, changes system settings, modifies registry, formats disks, writes to system folders),
  set risk="high" and propose a safer alternative or ask for clarification in the explanation while STILL providing one command
  that gathers information instead of doing damage.
- If you believe the task is complete, set done=true, command=null, and provide done_summary + next_goal_prompt.

Interaction:
- You will receive JSON feedback about the last command's stdout/stderr/exit_code.
- Use that feedback to decide the next command.

Keep explanations brief.
""".strip()


def build_user_payload(goal: str, shell_name: str) -> Dict[str, Any]:
    return {
        "type": "goal",
        "goal": goal,
        "shell": shell_name,
        "timestamp": int(time.time()),
    }


def build_feedback_payload(exec_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "command_result",
        "result": exec_result,
        "timestamp": int(time.time()),
    }


def call_model(
    client: OpenAI,
    cfg: AppConfig,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
) -> Dict[str, Any]:
    # Responses API with structured outputs in text.format. :contentReference[oaicite:4]{index=4}
    resp = client.responses.create(
        model=cfg.model,
        reasoning={"effort": cfg.reasoning_effort},
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": json_schema["name"],
                "strict": True,
                "schema": json_schema["schema"],
            }
        },
    )
    raw = extract_output_text(resp)
    return parse_step_json(raw)


# -----------------------------
# UI / Loop
# -----------------------------

def prompt_yes_edit_skip(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Returns: ("yes"|"edit"|"skip"|"quit", edited_command_if_any)
    """
    print(prompt)
    print("Confirm? [y]es / [e]dit / [s]kip / [q]uit")
    while True:
        choice = input("> ").strip().lower()
        if choice in ("y", "yes"):
            return ("yes", None)
        if choice in ("e", "edit"):
            new_cmd = input("Enter edited command:\n> ")
            return ("edit", new_cmd)
        if choice in ("s", "skip"):
            return ("skip", None)
        if choice in ("q", "quit"):
            return ("quit", None)
        print("Invalid input. Use y/e/s/q.")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python smart_shell_agent.py path/to/config.yml")
        return 2

    cfg = load_config(sys.argv[1])

    client_kwargs: Dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    client = OpenAI(**client_kwargs)  # SDK quickstart. :contentReference[oaicite:5]{index=5}

    shell_name, shell_prefix = detect_shell()
    schema = build_json_schema()

    print(f"Shell detected: {shell_name}")
    print("Enter a goal (empty line to exit).")

    while True:
        goal = input("\nGOAL> ").strip()
        if not goal:
            break

        # Conversation for this goal
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_instructions(shell_name)},
            {"role": "user", "content": json.dumps(build_user_payload(goal, shell_name), ensure_ascii=False)},
        ]

        while True:
            step = call_model(client, cfg, messages, schema)

            if cfg.show_raw_model_json:
                print("\n[Model JSON]")
                print(json.dumps(step, indent=2, ensure_ascii=False))

            # Done?
            if step.get("done") is True:
                print("\nDONE")
                if step.get("done_summary"):
                    print(step["done_summary"])
                if step.get("next_goal_prompt"):
                    print(step["next_goal_prompt"])
                break

            cmd = step.get("command") or ""
            target_shell = step.get("shell")

            # Enforce shell match
            if target_shell != shell_name:
                print(f"\nModel proposed shell={target_shell}, but host shell={shell_name}. Blocking.")
                messages.append({"role": "user", "content": json.dumps({
                    "type": "policy_violation",
                    "error": "shell_mismatch",
                    "host_shell": shell_name,
                    "model_shell": target_shell,
                }, ensure_ascii=False)})
                continue

            # Safety checks
            denied_by = is_denied(cmd, cfg.denylist_regex)
            allowed = is_allowed_by_prefix(cmd, cfg.allowlist_prefixes)
            if denied_by or (cfg.require_allowlist and not allowed):
                print("\nBLOCKED (safety policy)")
                if denied_by:
                    print(f"- Matched denylist regex: {denied_by}")
                if cfg.require_allowlist and not allowed:
                    print("- Not in allowlist prefixes")
                print(f"Proposed command: {cmd}")

                # Ask model for a safer inspection command
                messages.append({"role": "user", "content": json.dumps({
                    "type": "safety_block",
                    "blocked_command": cmd,
                    "denylist_match": denied_by,
                    "require_allowlist": cfg.require_allowlist,
                    "allowlist_prefixes": cfg.allowlist_prefixes,
                    "instruction": "Provide a safer alternative command (prefer read-only inspection).",
                }, ensure_ascii=False)})
                continue

            # Present to user
            print("\nNEXT STEP")
            print(f"Explanation: {step.get('explanation','')}")
            print(f"Risk: {step.get('risk','low')}")
            print(f"Expected outcome: {step.get('expected_outcome','')}")
            print(f"\nCommand:\n{cmd}\n")

            action, edited = prompt_yes_edit_skip(step.get("confirmation_prompt", "Run this command?"))
            if action == "quit":
                return 0
            if action == "skip":
                messages.append({"role": "user", "content": json.dumps({
                    "type": "user_skipped",
                    "skipped_command": cmd,
                    "reason": "user_selected_skip",
                }, ensure_ascii=False)})
                continue

            final_cmd = edited if action == "edit" else cmd

            # Run
            exec_result = run_command(
                shell_name=shell_name,
                shell_prefix=shell_prefix,
                command=final_cmd,
                timeout_seconds=cfg.timeout_seconds,
                max_output_chars=cfg.max_output_chars,
                workdir=cfg.workdir,
                extra_env=cfg.env,
            )

            # Show result to console
            print("\n[EXEC RESULT]")
            print(json.dumps(exec_result, indent=2, ensure_ascii=False))

            # Send feedback to model
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            messages.append({"role": "user", "content": json.dumps(build_feedback_payload(exec_result), ensure_ascii=False)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
