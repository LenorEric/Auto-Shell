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
from openai.types.shared_params import Reasoning


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


def dump_list_dict_str(data: list[dict[str, str]]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


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


_RE_NBSP = re.compile(r"[\u00A0\u2007\u202F]")  # common non-breaking spaces
_RE_HSPACE = re.compile(r"[ \t\u2000-\u200A\u205F]+")  # horizontal whitespace (space-like)
_RE_TRAIL_SPACE = re.compile(r"[ \t]+(?=\r?\n)")  # trailing spaces before newline
_RE_MANY_BLANKS = re.compile(r"(?:\r?\n){3,}")  # 3+ newlines


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
                "cmd_type": {"type": "string", "enum": ["done", "python", "shell", "query"],
                         "description": "Current step type. If all works are done, set to 'done'. If proposing a shell command, set to 'shell'. If proposing a python code snippet, set to 'python'. If asking for more info in text, set to 'query'."},
                "command": {"type": ["string", "null"],
                            "description": "Exactly one shell command to be directly run if type=shell, or question that needs ensure from the user if type=query, or null if done=true. Multiple commands shall be joined with proper connector if necessary to save steps."},
                "python_code" : {"type": ["string", "null"],
                                    "description": "Python code snippet to be executed if type=python, or null otherwise."},
                "explanation": {"type": "string",
                                "description": "Analyze this command and its parameters from a technical perspective, and briefly explain its function. Or null if type=query"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"],
                         "description": "Risk level of current command to be run(just focus on the current one, not the global intention). Or null if type=query"},
                "conclusion_prompt": {"type": "string",
                                        "description": "Before the user decides whether to execute it, explain to the user the overall effect of this command and the reason for doing so.  Or null if type=query"},
                "expected_outcome": {"type": "string", "description": "What success console output looks like for this step's command. Or null if type=query"},
                "done_summary": {"type": ["string", "null"],
                                 "description": "If done=true, summarize what was accomplished or give final answer."},
            },
            "required": ["cmd_type", "command", "python_code", "explanation", "risk", "conclusion_prompt", "expected_outcome",
                         "done_summary"],
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
You are SmartShell, an assistant that completes a user's goal by proposing ONE action at a time (shell command, python script, or query).

Hard rules:
- Output MUST be valid JSON matching the provided schema (no markdown, no extra keys).
- Propose EXACTLY ONE action per step by setting cmd_type to "shell", "python", "query", or "done".
- If the goal is vague or requires clarification/choice, set cmd_type="query" and put your question in the "command" field. Set fields like risk, explanation, and expected_outcome to null.
- If a Python script is more suitable than shell commands, set cmd_type="python" and provide the code in "python_code" (no comments needed in the script).
- Shell commands MUST target the user's shell: "{shell_name}" only.
- Prefer safe, read-only inspection commands.
- If a "shell" or "python" step is risky (deletes data, changes settings, modifies registry and so on), set risk="high" and suggest an alternative in the conclusion_prompt. Risk level refers only to the current command.
- You may receive a JSON "supplement" when the user adds constraints. Treat it as a high-priority update and adapt the next step.
- If the user rejects a command, do not insist on repeating it; adapt and try another way.
- If the task is complete, set cmd_type="done", command=null, and provide done_summary.

Interaction:
- You will receive JSON feedback about the last command's stdout/stderr/exit_code (or python execution results) or user's answer.
- Use that feedback to decide the next action.

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


def build_supplement_payload(rejected_command: str, supplement: str) -> Dict[str, Any]:
    return {
        "type": "supplement",
        "rejected_command": rejected_command,
        "supplement": supplement,
        "timestamp": int(time.time()),
    }


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
            reasoning=eval_reasoning_effort(cfg.reasoning_effort),
            input=dump_list_dict_str(temp_messages),
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
        reasoning=eval_reasoning_effort(cfg.reasoning_effort),
        input=dump_list_dict_str(messages),
        text={"format": {"type": "json_schema", "name": json_schema["name"], "strict": True,
                         "schema": json_schema["schema"], }},
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
      ("run" | "edit" | "reject" | "ask" | "instruct", optional_text)

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

        if choice in ("r", "reject"):
            reason = input("Enter a reason (leave blank for no reason):\n> ")
            return "reject", reason

        if choice in ("a", "ask"):
            return "ask", None

        if choice in ("i", "instruct"):
            supp = input("Supplementary instruction for the model:\n> ")
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

    def serve_shell_cmd():
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
            return 1

        # Present to user
        print(f"\nCommand:\n{cmd}\n")
        print(f"Explanation: {step.get('explanation', '')}")
        print(f"Expected outcome: {step.get('expected_outcome', '')}")
        print(f"Risk: {step.get('risk', 'UNKNOWN')}\n")

        # --- replace the old action handling block with this ---

        _skip_lvl = 0
        final_cmd = ""
        while True:
            action, payload = prompt_action(step.get("conclusion_prompt", "Run this command?"))

            if action == "ask":
                interjection_qa(client, cfg, shell_name, cmd)
                # IMPORTANT: do NOT attempt to interpret payload; just loop and prompt again.
                continue

            if action == "reject":
                messages.append({"role": "user", "content": json.dumps({
                    "type": "user_rejected",
                    "user_rejected": cmd,
                    "reason": payload or "user rejected without reason",
                }, ensure_ascii=False)})
                # skip this command and request next model step
                _skip_lvl = 1
                break

            if action == "instruct":
                supplement = (payload or "").strip()
                messages.append({"role": "user", "content": json.dumps(
                    build_supplement_payload(cmd, supplement),
                    ensure_ascii=False
                )})
                # skip this command and request next model step (with supplement)
                _skip_lvl = 1
                break

            if action == "run":
                final_cmd = cmd
                _skip_lvl = 0
                break

            if action == "edit":
                final_cmd = (payload or cmd)
                _skip_lvl = 0
                break

            if action == "quit":
                _skip_lvl = 2
                break

        # After the loop:
        if _skip_lvl:
            return _skip_lvl

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
            {"role": "user",
             "content": json.dumps(build_feedback_payload(exec_result, cfg.compress_for_llm), ensure_ascii=False)})

        return 0

    def serve_python_cmd():
        return 0

    def serve_query_cmd():
        return 0

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

    if not (is_admin()):
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
            print(f"\n\n**********\nSTEP {step_cnt}")

            if cfg.show_raw_model_json:
                print("\n[Model JSON]")
                print(json.dumps(step, indent=2, ensure_ascii=False))

            skip_lvl = 0

            # Done?
            if step.get("cmd_type") == "done":
                print("\nDONE")
                if step.get("done_summary"):
                    print(step["done_summary"])
                skip_lvl = 2
            elif step.get("cmd_type") == "shell":
                skip_lvl = serve_shell_cmd()
            elif step.get("cmd_type") == "python":
                skip_lvl = serve_python_cmd()
            elif step.get("cmd_type") == "query":
                skip_lvl = serve_query_cmd()

            if skip_lvl == 2:
                break


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
