#!/usr/bin/env python3
"""
Requires:
  pip install openai pyyaml
"""
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
    exec_preview_chars: int
    compress_for_llm: bool
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
        max_output_chars=int(agent_cfg.get("max_output_chars", 8192)),
        exec_preview_chars=int(agent_cfg.get("exec_preview_chars", 2048)),
        compress_for_llm=bool(safety_cfg.get("compress_for_llm", True)),
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


_RE_NBSP = re.compile(r"[\u00A0\u2007\u202F]")          # common non-breaking spaces
_RE_HSPACE = re.compile(r"[ \t\u2000-\u200A\u205F]+")   # horizontal whitespace (space-like)
_RE_TRAIL_SPACE = re.compile(r"[ \t]+(?=\r?\n)")        # trailing spaces before newline
_RE_MANY_BLANKS = re.compile(r"(?:\r?\n){3,}")          # 3+ newlines

def compress_for_llm(
    s: Optional[str],
    *,
    keep_indentation: bool = True,
    max_consecutive_blank_lines: int = 2,
    strip_ends: bool = True,
) -> str:
    """
    LLM-friendly whitespace compression:
    - Normalizes NBSP-like chars to plain space
    - Removes trailing spaces at line ends
    - Collapses repeated horizontal whitespace inside lines
      (optionally preserving leading indentation)
    - Limits consecutive blank lines (default: 2)
    """
    if not s:
        return ""

    # Normalize line endings and NBSP-like spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_NBSP.sub(" ", s)

    # Remove trailing spaces at end of lines
    s = _RE_TRAIL_SPACE.sub("", s)

    lines = s.split("\n")
    out_lines = []

    for line in lines:
        if keep_indentation:
            # Preserve leading indentation (spaces/tabs) exactly, compress the rest
            m = re.match(r"^[ \t]+", line)
            indent = m.group(0) if m else ""
            rest = line[len(indent):]
            rest = _RE_HSPACE.sub(" ", rest).strip(" ")
            out_lines.append(indent + rest)
        else:
            # Compress all horizontal whitespace, then trim line edges
            out_lines.append(_RE_HSPACE.sub(" ", line).strip(" "))

    s = "\n".join(out_lines)

    # Limit consecutive blank lines
    if max_consecutive_blank_lines is not None:
        n = max(0, int(max_consecutive_blank_lines))
        s = re.sub(r"\n{" + str(n + 2) + r",}", "\n" * (n + 1), s)

    if strip_ends:
        s = s.strip()

    return s


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
                "type": {"type": "string", "enum": ["done", "python", "shell", "query"],
                         "description": "Current step type. If all works are done, set to 'done'. If proposing a shell command, set to 'shell'. If proposing a python code snippet, set to 'python'. If asking for more info in text, set to 'query'."},
                "command": {"type": ["string", "null"],
                            "description": "Exactly one shell command to run, or question that needs ensure from the user, or null if done=true."},
                "explanation": {"type": "string",
                                "description": "Analyze this command and its parameters from a technical perspective, and briefly explain its function."},
                "risk": {"type": "string", "enum": ["low", "medium", "high"],
                         "description": "Risk level of current command to be run(just focus on the current one, not the global intention)."},
                "confirmation_prompt": {"type": "string",
                                        "description": "Before the user decides whether to execute it, explain to the user the overall effect of this command and the reason for doing so."},
                "expected_outcome": {"type": "string", "description": "What success looks like for this step."},
                "done_summary": {"type": ["string", "null"],
                                 "description": "If done=true, summarize what was accomplished."},
                "next_goal_prompt": {"type": ["string", "null"],
                                     "description": "If done=true, ask user what to do next."},
            },
            "required": ["type", "command", "explanation", "risk", "confirmation_prompt", "expected_outcome",
                         "done_summary", "next_goal_prompt"],
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
- If current step is risky (deletes data, changes system settings, modifies registry, formats disks, writes to system folders),
  set risk="high" and suggest user another alternative command in the confirmation_prompt. 
  Notice that risk level only refers to the current command, not the overall goal.
- You may receive a JSON message of type "supplement" when the user skips a command and adds constraints.
  Treat it as a high-priority update to the plan, and propose the next best single command accordingly.
- If the user skips a command, do not insist on repeating it; adapt.

- If you believe the task is complete, set done=true, command=null, and provide done_summary + next_goal_prompt.

Interaction:
- You will receive JSON feedback about the last command's stdout/stderr/exit_code.
- Use that feedback to decide the next command.

Keep explanations brief.
""".strip()


def build_interjection_system_instructions(shell_name: str) -> str:
    return f"""
You are a command explainer for a user reviewing a proposed shell command.

Context:
- Shell: {shell_name}
- The user will ask questions about the command’s meaning, risks, and what it changes.
- Do NOT propose alternative commands.
- Do NOT suggest multi-step plans.
- Keep answers short, concrete, and focused on safety and effect.
- Output in simple PLAIN text without markdown format because the interaction occurs in the terminal console.
- If the command is dangerous or ambiguous, clearly state why and what to verify before running.
""".strip()


def build_user_payload(goal: str, shell_name: str) -> Dict[str, Any]:
    return {
        "type": "goal",
        "goal": goal,
        "shell": shell_name,
        "timestamp": int(time.time()),
    }


def build_feedback_payload(exec_result: Dict[str, Any], compress: bool) -> Dict[str, Any]:
    if compress:
        exec_result["stdout"] = compress_for_llm(exec_result.get("stdout", ""))
        exec_result["stderr"] = compress_for_llm(exec_result.get("stderr", ""))
    return {
        "type": "command_result",
        "result": exec_result,
        "timestamp": int(time.time()),
    }


def build_supplement_payload(skipped_command: str, supplement: str) -> Dict[str, Any]:
    return {
        "type": "supplement",
        "skipped_command": skipped_command,
        "supplement": supplement,
        "timestamp": int(time.time()),
    }


def interjection_qa(
        client: OpenAI,
        cfg: AppConfig,
        shell_name: str,
        command: str,
) -> None:
    """
    Inline Q&A about a proposed command.
    This does NOT modify the main goal conversation.
    End by entering a single '.' (dot) on its own line.
    """
    print("\nASK MODE (end with a single '.' line)")
    while True:
        q = input("ASK> ").strip()
        if q == ".":
            print("Exit ASK MODE.\n")
            return
        if not q:
            continue

        temp_messages = [
            {"role": "system", "content": build_interjection_system_instructions(shell_name)},
            {"role": "user",
             "content": json.dumps({"type": "proposed_command", "shell": shell_name, "command": command},
                                   ensure_ascii=False)},
            {"role": "user", "content": q},
        ]

        # Plain text reply is best here (no schema).
        resp = client.responses.create(
            model=cfg.model,
            reasoning={"effort": cfg.reasoning_effort},
            input=temp_messages,
        )
        answer = extract_output_text(resp).strip()
        print(answer + "\n")


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
def format_shell_text(s: str) -> str:
    """
    If s contains literal backslash-n sequences (\\n) instead of real newlines,
    convert them. Also normalize CRLF.
    """
    if not s:
        return ""
    # Convert literal "\n" to actual newline only if it looks like it was escaped.
    # This avoids double-transforming already-normal text.
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    # Normalize actual CRLF to LF for consistent console output
    s = s.replace("\r\n", "\n")
    return s


def prompt_action(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Returns:
      ("run" | "edit" | "skip" | "ask" | "instruct", optional_text)

    - edit: optional_text = edited command
    - instruct: optional_text = supplementary instruction
    """
    print(prompt)
    print("Choose: [y]es(run) / [e]dit+run / [s]kip / [a]sk / [i]nstruct / [q]uit")
    while True:
        choice = input("> ").strip().lower()

        if choice in ("y", "yes"):
            return "run", None

        if choice in ("e", "edit"):
            new_cmd = input("Enter edited command:\n> ")
            return "edit", new_cmd

        if choice in ("s", "skip"):
            return "skip", None

        if choice in ("a", "ask"):
            return "ask", None

        if choice in ("i", "instruct"):
            supp = input("Supplementary instruction for the model (why skip / what to do instead):\n> ")
            return "instruct", supp

        if choice in ("q", "quit", "exit"):
            return "quit", None

        print("Invalid input. Use y/e/s/a/i.")


def main() -> int:
    def is_admin() -> bool:
        if os.name == "nt":  # Windows
            try:
                import ctypes
                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        else:  # Linux / macOS / Unix
            return os.geteuid() == 0

    def clip_text(s: str, limit: int) -> str:
        if not s:
            return ""
        count = 0
        end_idx = 0
        for i, ch in enumerate(s):
            if not ch.isspace():
                count += 1
            if count > limit:
                break
            end_idx = i + 1
        if count <= limit:
            return s
        return s[:end_idx] + "\n...[truncated]"

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

    if not(is_admin()):
        print("WARNING: The agent is NOT running with administrative/root privileges.")
        print("Some commands may fail due to insufficient permissions. You may want to elevate your privileges.\n")
    print(f"Shell detected: {shell_name}")
    print("Enter a goal (empty line to exit).")

    while True:
        goal = input("\nGOAL> ").strip()
        if not goal:
            break

        step_cnt = 0

        # Conversation for this goal
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_instructions(shell_name)},
            {"role": "user", "content": json.dumps(build_user_payload(goal, shell_name), ensure_ascii=False)},
        ]

        while True:
            step = call_model(client, cfg, messages, schema)
            step_cnt += 1

            if cfg.show_raw_model_json:
                print("\n[Model JSON]")
                print(json.dumps(step, indent=2, ensure_ascii=False))

            # Done?
            if step.get("type") == "done":
                print("\nDONE")
                if step.get("done_summary"):
                    print(step["done_summary"])
                if step.get("next_goal_prompt"):
                    print(step["next_goal_prompt"])
                break

            cmd = step.get("command") or ""

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
            print(f"\n\n**********\nSTEP {step_cnt}")
            print(f"\nCommand:\n{cmd}\n")
            print(f"Explanation: {step.get('explanation', '')}")
            print(f"Expected outcome: {step.get('expected_outcome', '')}")
            print(f"Risk: {step.get('risk', 'UNKNOWN')}\n")

            # --- replace the old action handling block with this ---

            skip_level = False
            final_cmd = ""
            while True:
                action, payload = prompt_action(step.get("confirmation_prompt", "Run this command?"))

                if action == "ask":
                    interjection_qa(client, cfg, shell_name, cmd)
                    # IMPORTANT: do NOT attempt to interpret payload; just loop and prompt again.
                    continue

                if action == "skip":
                    messages.append({"role": "user", "content": json.dumps({
                        "type": "user_skipped",
                        "skipped_command": cmd,
                        "reason": "user_selected_skip",
                    }, ensure_ascii=False)})
                    # skip this command and request next model step
                    skip_level = 1
                    break

                if action == "instruct":
                    supplement = (payload or "").strip()
                    messages.append({"role": "user", "content": json.dumps(
                        build_supplement_payload(cmd, supplement),
                        ensure_ascii=False
                    )})
                    # skip this command and request next model step (with supplement)
                    skip_level = 1
                    break

                if action == "run":
                    final_cmd = cmd
                    skip_level = 0
                    break

                if action == "edit":
                    final_cmd = (payload or cmd)
                    skip_level = 0
                    break

                if action == "quit":
                    skip_level = 2
                    break

            # After the loop:
            if skip_level == 2:
                break
            if skip_level == 1:
                continue

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
            preview_limit = cfg.exec_preview_chars

            stdout_text = clip_text(format_shell_text(exec_result.get("stdout", "") or ""), preview_limit)
            stderr_text = clip_text(format_shell_text(exec_result.get("stderr", "") or ""), preview_limit)

            print("\n[EXEC RESULT]")
            print(f"shell: {exec_result.get('shell')}")
            print(f"cwd: {exec_result.get('cwd')}")
            print(f"exit_code: {exec_result.get('exit_code')}")
            print(f"timed_out: {exec_result.get('timed_out')}")
            print(f"elapsed_seconds: {exec_result.get('elapsed_seconds')}")

            print("\n[STDOUT]")
            if stdout_text:
                print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
            else:
                print("(empty)")

            if stderr_text:
                print("\n[STDERR]")
                print(stderr_text, end="" if stderr_text.endswith("\n") else "\n")

            # Send feedback to model
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            messages.append(
                {"role": "user", "content": json.dumps(build_feedback_payload(exec_result, cfg.compress_for_llm), ensure_ascii=False)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
