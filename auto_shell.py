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
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    selected_model: str
    reasoning_effort: str
    timeout_seconds: int
    max_output_chars: int
    exec_preview_chars: int
    compress_for_llm: bool
    workdir: Optional[str]
    env: Dict[str, str]
    show_raw_model_json: bool


def load_config(path: str) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    openai_cfg = data.get("openai", {})
    agent_cfg = data.get("agent", {})

    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key (set openai.api_key or OPENAI_API_KEY).")

    return AppConfig(
        api_key=api_key,
        base_url=openai_cfg.get("base_url"),
        model=openai_cfg.get("model", "auto"),
        selected_model=openai_cfg.get("model", "auto"),
        reasoning_effort=openai_cfg.get("reasoning_effort", "low"),
        timeout_seconds=int(agent_cfg.get("timeout_seconds", 30)),
        max_output_chars=int(agent_cfg.get("max_output_chars", 8192)),
        exec_preview_chars=int(agent_cfg.get("exec_preview_chars", 2048)),
        compress_for_llm=bool(agent_cfg.get("compress_for_llm", True)),
        workdir=agent_cfg.get("workdir"),
        env={str(k): str(v) for k, v in (agent_cfg.get("env", {}) or {}).items()},
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
                "operation": {"type": ["string", "null"],
                              "description": "Exactly one shell command (multiple commands shall be joined with proper connector if necessary to save steps) to be directly run if type=shell, or python code snippet to be executed if type=python, or question that needs ensure from the user if type=query, or null if done=true."},
                "explanation": {"type": "string",
                                "description": "Analyze this command and its parameters from a technical perspective, and briefly explain its function. Or null if type=query"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"],
                         "description": "Risk level of current command to be run(just focus on the current one, not the global intention). Or null if type=query"},
                "conclusion_prompt": {"type": "string",
                                      "description": "Before the user decides whether to execute it, explain to the user the overall effect of this command and the reason for doing so.  Or null if type=query"},
                "expected_outcome": {"type": "string",
                                     "description": "What success console output looks like for this step's command. Or null if type=query"},
                "done_summary": {"type": ["string", "null"],
                                 "description": "If done=true, summarize what was accomplished or give final answer."},
            },
            "required": ["cmd_type", "operation", "explanation", "risk", "conclusion_prompt", "expected_outcome",
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
    def extract_first_json(text: str) -> str | None:
        start = text.find('{')
        if start == -1:
            return None

        brace_count = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i + 1]

        return None

    try:
        return json.loads(raw_text)
    except Exception:
        try:
            fixed_text = extract_first_json(raw_text)
            if fixed_text:
                loaded = json.loads(fixed_text)
                print(
                    "[WARN] Error occurred when parsing model output as JSON, but successfully extracted JSON portion.")
                return loaded
        except Exception:
            print("[ERROR] Failed to parse model output as JSON. Raw output:")
            print(raw_text)
            return {}


def build_system_instructions(shell_name: str) -> str:
    return f"""
You are SmartShell, an assistant that completes a user's goal by proposing ONE action at a time (shell command, python script, or query).

Hard rules:
- Output MUST be valid JSON matching the provided schema (no markdown, no extra keys).
- Propose EXACTLY ONE action per step by setting cmd_type to "shell", "python", "query", or "done".
- If the goal is vague or ambiguous (no need to double-check the clearly stated goal even if it's dangerous), or require user external operation (like asking for reboot), set cmd_type="query" and put your question in the "operation" field. Set risk=null, explanation=null, expected_outcome=null.
- Shell commands MUST target the user's shell: "{shell_name}" only.
- Shell should be used first, but if a Python script is more suitable, set cmd_type="python" and provide a concise and efficient Python code. (no comments needed in the script).
- The Python code MUST be compatible with Python version {platform.python_version()}.
- Python code MUST include a `handle()` function as its entry point. At runtime, only this function will be executed as the entry point. The return value of this function will feedback to you.
- Python code MUST NOT containing console input calls (e.g., input()), and try to avoid output as possible.
- Python code SHALL use standard libraries, or confirm the libraries are pre-installed before used.
- Prefer safe, read-only inspection commands.
- If a "shell" or "python" step is risky (deletes data, changes settings, modifies registry and so on), set risk="high" and suggest an alternative in the conclusion_prompt. Risk level refers only to the current command.
- You may receive a JSON "supplement" when the user adds constraints. Treat it as a high-priority update and adapt the next step.
- If the user rejects a command, do not insist on repeating it; adapt and try another way.
- If reboot (or leave) is needed, ask the user via a "query" step, don't directly reboot.
- If the task is complete, set cmd_type="done", operation=null, and provide done_summary.

Interaction:
- You will receive JSON feedback about the last command's stdout/stderr/exit_code (or python execution results) or user's answer.
- Use that feedback to decide the next action.

Keep explanations brief.
""".strip()


def build_interjection_system_instructions(shell_name: str) -> str:
    return f"""
You are a command explainer for a user reviewing a proposed shell command from smart shell assistant.

Context:
- Shell: {shell_name} / Python Version: {platform.python_version()}
- The history context between user and smart shell assistant is given below, after that, it's the user's question.
- The user will ask questions about the given command.
- Do NOT propose alternative commands or plans.
- Keep answers short, concrete, and focused on safety and effect.
- Output in simple PLAIN text without markdown format because the interaction occurs in the terminal console.
- If the command is dangerous or ambiguous, clearly state why and what to verify before running.
""".strip()


def build_user_payload(goal: str, shell_name: str) -> Dict[str, Any]:
    return {
        "act_type": "goal",
        "goal": goal,
        "shell": shell_name,
        "python_version": platform.python_version(),
        "timestamp": int(time.time()),
    }


def build_cmd_feedback_payload(exec_result: Dict[str, Any], compress: bool) -> Dict[str, Any]:
    if compress:
        exec_result["stdout"] = compress_for_llm(exec_result.get("stdout", ""))
        exec_result["stderr"] = compress_for_llm(exec_result.get("stderr", ""))
    return {
        "act_type": "command_result",
        "result": exec_result,
        "timestamp": int(time.time()),
    }


def build_query_feedback_payload(answer: str) -> Dict[str, Any]:
    return {
        "act_type": "query_answer",
        "answer": answer,
    }


def build_supplement_payload(supplement: str) -> Dict[str, Any]:
    return {
        "act_type": "supplement",
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
        context: List[Dict[str, str]],
) -> None:
    """
    Inline Q&A about a proposed command.
    This does NOT modify the main goal conversation.
    End by entering a single '.' (dot) on its own line.
    """
    context_messages = context[1:]

    temp_messages = [
        {"role": "system", "content": build_interjection_system_instructions(shell_name)},
        *context_messages,
        {"role": "user",
         "content": "The above is the historical operation record; the following are the user's questions."},
        {"role": "user",
         "content": json.dumps({"act_type": "proposed_command", "command": command},
                               ensure_ascii=False)}
    ]

    print("\nASK MODE ( (empty line to exit))")
    while True:
        q = input("ASK> ").strip()
        if not q:
            print("Exit ASK MODE.\n")
            return
        if not q:
            continue

        temp_messages.append({"role": "user", "content": q})
        resp = client.responses.create(
            model=cfg.selected_model,
            # reasoning=eval_reasoning_effort(cfg.reasoning_effort),
            input=temp_messages,
        )
        temp_messages.append({"role": "assistant", "content": extract_output_text(resp)})
        answer = extract_output_text(resp).strip()
        print(answer + "\n")


def call_model(
        client: OpenAI,
        cfg: AppConfig,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
) -> Dict[str, Any]:
    while True:
        resp = client.responses.create(
            model=cfg.selected_model,
            # reasoning=eval_reasoning_effort(cfg.reasoning_effort),
            input=messages,
            text={"format": {"type": "json_schema", "name": json_schema["name"], "strict": True,
                             "schema": json_schema["schema"]}},
        )
        raw = extract_output_text(resp)
        parsed = parse_step_json(raw)
        if parsed and "cmd_type" in parsed:
            break
        print("[WARNING] Model output missing required 'cmd_type'. Retrying after 3s...\n")
        time.sleep(3)
    return parsed


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
    print("Choose: [y]es(run) / [r]eject / [a]sk / [i]nstruct / [q]uit")
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
            if not supp.strip():
                print("Supplement cannot be empty.")
                continue
            else:
                return "instruct", supp

        if choice in ("q", "quit", "exit"):
            return "quit", None

        print("Invalid input. Use y/e/r/a/i.")


def dump_history(
        messages: Union[list, dict],
        step_cnt: int,
        selected_model: str,
        path: Union[str, Path] = "history.json"
) -> None:
    data = {
        "messages": messages,
        "step_cnt": int(step_cnt),
        "selected_model": selected_model
    }

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_history(
        path: Union[str, Path] = "history.json"
) -> tuple[Union[list, dict], int, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages")
    step_cnt = int(data.get("step_cnt", 0))
    selected_model = data.get("selected_model", "gpt-5-chat-latest")

    return messages, step_cnt, selected_model


import json
from typing import List, Dict, Any


def replay_console_history(messages: List[Dict[str, str]]) -> None:
    """
    Rebuild and print historical console output from saved messages.
    This does NOT execute commands; it only replays what was previously shown.
    """

    step_idx = 0
    last_step = None

    def _print_exec_result(result: Dict[str, Any]) -> None:
        print("\n[EXEC RESULT]")
        print(f"cwd: {result.get('cwd')}")
        print(f"exit_code: {result.get('exit_code')}")
        print(f"timed_out: {result.get('timed_out')}")
        print(f"elapsed_seconds: {result.get('elapsed_seconds')}")

        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""

        print("\n[STDOUT]")
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        else:
            print("(empty)")

        if stderr:
            print("\n[STDERR]")
            print(stderr, end="" if stderr.endswith("\n") else "\n")

    for msg in messages[:-1]:
        role = msg.get("role")
        content = msg.get("content")

        if not content:
            continue

        # Try JSON decode; skip plain text system messages
        try:
            payload = json.loads(content)
        except Exception:
            continue

        # ---- Assistant step ----
        if role == "assistant" and isinstance(payload, dict) and "cmd_type" in payload:
            step_idx += 1

            print(f"\n\n**********")
            print(f"STEP {step_idx}")

            cmd_type = payload.get("cmd_type")

            if cmd_type == "shell":
                print(f"\nCommand:\n{payload.get('operation')}\n")
                print(f"Explanation: {payload.get('explanation')}")
                print(f"Expected outcome: {payload.get('expected_outcome')}")
                print(f"Risk: {payload.get('risk')}\n")

            elif cmd_type == "python":
                print(f"\nCommand:\n{payload.get('operation')}\n")
                print(f"Explanation: {payload.get('explanation')}")
                print(f"Expected outcome: {payload.get('expected_outcome')}")
                print(f"Risk: {payload.get('risk')}\n")

            elif cmd_type == "query":
                print(f"\nQuery:\n{payload.get('operation')}\n")

            elif cmd_type == "done":
                if payload.get("done_summary"):
                    print(payload["done_summary"])
                print("DONE\n**********\n")

        # ---- User feedback ----
        elif role == "user" and isinstance(payload, dict) and "act_type" in payload:
            ptype = payload.get("act_type")

            if ptype == "goal":
                print(f"\n\nGOAL> {payload.get('goal')}")

            elif ptype == "command_result":
                result = payload.get("result", {})
                _print_exec_result(result)

            elif ptype == "query_answer":
                print(f"ANS> {payload.get('answer')}")

            elif ptype == "supplement":
                print("\n[SUPPLEMENT]")
                print(payload.get("supplement"))

            elif ptype == "user_rejected":
                print("\n[USER REJECTED]")
                print(payload.get("user_rejected"))
                if payload.get("reason"):
                    print(f"Reason: {payload.get('reason')}")


def auto_select_model(client: OpenAI, goal: str) -> tuple[str, str]:
    router_model = "gpt-4.1-mini"

    model_map = {
        "large": "gpt-5-chat-latest",
        "medium": "gpt-4.1-mini",
        "small": "gpt-4.1-nano",
    }

    system_prompt = """
        You are a task complexity evaluator for a computer automation agent engine.

        Given a user's goal, classify the task complexity into exactly one of:
        - "small": very explicit, short, deterministic operations; minimal steps; low risk.
        - "medium": clear intent with multiple steps or rules; some branching; moderate length.
        - "large": ambiguous, diagnostic, open-ended, or long-running tasks; requires troubleshooting, system understanding, or architectural decisions.

        Rules:
        1. If the complexity is uncertain, choose the larger.
        2. Do not solve the task.
        3. Base your decision on ambiguity, number of steps, need for diagnosis, and risk.
        4. Output must be valid JSON only.
        """.strip()

    text_schema = {
        "format": {
            "type": "json_schema",
            "name": "task_complexity_evaluator",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "complexity": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "Evaluated task complexity level"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score of the evaluation"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief technical justification for the selected complexity"
                    }
                },
                "required": ["complexity", "confidence", "reason"]
            }
        }
    }

    response = client.responses.create(
        model=router_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ],
        text=text_schema
    )

    # 提取文本输出
    output_text = response.output_text.strip()

    try:
        result = json.loads(output_text)
        complexity = result.get("complexity")
        reason = result.get("reason", "")
    except json.JSONDecodeError as e:
        reason = ""
        print(output_text)
        print("[WARN] Router output parse failed; escalated to large model.", e)
        complexity = "large"

    if complexity not in model_map:
        print("[WARN] Router output parse failed; escalated to large model.")
        complexity = "large"

    return model_map[complexity], reason


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
        def run_shell_command(
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
                    "cwd": cwd,
                    "actual_run_cmd": final_cmd,
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
                    "cwd": cwd,
                    "actual_run_cmd": final_cmd,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "elapsed_seconds": round(elapsed, 3),
                }

        messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
        cmd = step.get("operation") or "No valid cmd got from llm"

        # Present to user
        print(f"\nCommand:\n{cmd}\n")
        print(f"Explanation: {step.get('explanation', '')}")
        print(f"Expected outcome: {step.get('expected_outcome', '')}")
        print(f"Risk: {step.get('risk', 'UNKNOWN')}\n")

        _skip_lvl = 0
        final_cmd = ""
        while True:
            action, payload = prompt_action(step.get("conclusion_prompt", "Run this command?"))

            if action == "ask":
                interjection_qa(client, cfg, shell_name, cmd, messages)
                continue

            if action == "reject":
                messages.append({"role": "user", "content": json.dumps({
                    "act_type": "user_rejected",
                    "user_rejected": cmd,
                    "reason": payload or "user rejected without reason",
                }, ensure_ascii=False)})
                # skip this command and request next model step
                _skip_lvl = 1
                break

            if action == "instruct":
                final_cmd = cmd
                supplement = (payload or "").strip()
                messages.append({"role": "user", "content": json.dumps(
                    build_supplement_payload(supplement),
                    ensure_ascii=False
                )})
                _skip_lvl = 0
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
        exec_result = run_shell_command(
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
        messages.append(
            {"role": "user",
             "content": json.dumps(build_cmd_feedback_payload(exec_result, cfg.compress_for_llm), ensure_ascii=False)})

        return 0

    def serve_python_cmd():
        def run_agent_code(code_str) -> Dict[str, Any]:
            try:
                if not isinstance(code_str, str):
                    raise TypeError(f"get_code() must return str, got {type(code_str).__name__}")

                exec_env = {"__builtins__": __builtins__}

                # Compile first so syntax errors are caught cleanly with traceback
                compiled = compile(code_str, "<agent_code>", "exec")
                exec(compiled, exec_env, exec_env)

                handle_func = exec_env.get("handle")
                if handle_func is None or not callable(handle_func):
                    raise RuntimeError("handle() function not found or not callable")

                result = handle_func()

                return {"ok": True, "result": result, "error": None}

            except Exception as e:
                return {
                    "ok": False,
                    "result": None,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }

        messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
        pyc = step.get("operation") or "No valid python code got from llm"

        # Present to user
        print(f"\nCommand:\n{pyc}\n")
        print(f"Explanation: {step.get('explanation', '')}")
        print(f"Expected outcome: {step.get('expected_outcome', '')}")
        print(f"Risk: {step.get('risk', 'UNKNOWN')}\n")

        _skip_lvl = 0
        while True:
            action, payload = prompt_action(step.get("conclusion_prompt", "Run this command?"))
            if action == "ask":
                interjection_qa(client, cfg, shell_name, pyc, messages)
                continue
            if action == "reject":
                messages.append({"role": "user", "content": json.dumps({
                    "act_type": "user_rejected",
                    "user_rejected": pyc,
                    "reason": payload or "user rejected without reason",
                }, ensure_ascii=False)})
                _skip_lvl = 1
                break
            if action == "instruct":
                supplement = (payload or "").strip()
                messages.append({"role": "user", "content": json.dumps(
                    build_supplement_payload(supplement),
                    ensure_ascii=False
                )})
                _skip_lvl = 1
                break
            if action == "edit":
                print("Editing Python code is not supported currently.")
                continue
            if action == "run":
                _skip_lvl = 0
                break
            if action == "quit":
                _skip_lvl = 2
                break

        # After the loop:
        if _skip_lvl:
            return _skip_lvl

        exec_result = run_agent_code(pyc)

        messages.append(
            {"role": "user",
             "content": json.dumps(build_cmd_feedback_payload(exec_result, cfg.compress_for_llm), ensure_ascii=False)})

        return 0

    def serve_query_cmd():
        qry = step.get("operation") or "No valid query got from llm"
        messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})

        # Present query to user
        print(f"\nQuery:\n{qry}\n")

        dump_history(messages, step_cnt, cfg.selected_model)

        answer = input("ANS> ")

        messages.append(
            {"role": "user",
             "content": json.dumps(build_query_feedback_payload(answer), ensure_ascii=False)})

        if hist_path.exists():
            hist_path.unlink()

        return 0

    def continue_query_cmd():
        replay_console_history(messages)
        print(f"\n\n**********\nSTEP {step_cnt}")
        qry = json.loads(messages[-1]["content"]).get("operation") or "No valid query got from history"
        print(f"\nQuery:\n{qry}\n")

        answer = input("ANS> ")

        messages.append(
            {"role": "user",
             "content": json.dumps(build_query_feedback_payload(answer), ensure_ascii=False)})

    if len(sys.argv) < 2:
        print("Usage: python smart_shell_agent.py path/to/config.yml")
        return 2

    shell_name, shell_prefix = detect_shell()
    if not (is_admin()):
        print("WARNING: The agent is NOT running with administrative/root privileges.")
        print("Some commands may fail due to insufficient permissions. You may want to elevate your privileges.\n")
    print(f"Shell detected: {shell_name}")
    print(f"Python version: {platform.python_version()}")
    print()

    cfg = load_config(sys.argv[1])

    client_kwargs: Dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    client = OpenAI(**client_kwargs)  # SDK quickstart. :contentReference[oaicite:5]{index=5}

    schema = build_json_schema()

    # If a previous history exists, load it so the agent can resume
    hist_path = Path("history.json")
    messages = None
    step_cnt = 0
    cfg.selected_model = "gpt-5-chat-latest"
    use_hist = False
    if hist_path.exists():
        use_hist = True if input("Found existing history.json. Load it? [y/n]: ").strip().lower() == "y" else False
        if use_hist:
            try:
                messages, step_cnt, cfg.selected_model = load_history(hist_path)
                print(f"Loaded history from `history.json` (step_cnt={step_cnt}).")
            except Exception as e:
                print(f"Failed to load history from `history.json`: {e}")
        else:
            print("Starting a new session.")
        hist_path.unlink()

    while True:
        if not use_hist:
            print("Enter a goal (empty line to exit).")
            goal = input("\nGOAL> ").strip()
            if not goal:
                break

            step_cnt = 0
            if cfg.model.lower() == "auto":
                cfg.selected_model, reason = auto_select_model(client, goal)
                print(f"\nUsing auto-selected model: {cfg.selected_model}")
                print(f"Router Reason: {reason}\n")
            else:
                cfg.selected_model = cfg.model

            # Conversation for this goal
            messages = [
                {"role": "system", "content": build_system_instructions(shell_name)},
                {"role": "user", "content": json.dumps(build_user_payload(goal, shell_name), ensure_ascii=False)},
            ]

        while True:
            if use_hist:
                continue_query_cmd()
                use_hist = False  # only load once at start

            step = call_model(client, cfg, messages, schema)
            step_cnt += 1
            print(f"\n\n**********\nSTEP {step_cnt}")

            if cfg.show_raw_model_json:
                print("\n[Model JSON]")
                print(json.dumps(step, indent=2, ensure_ascii=False))

            skip_lvl = 0

            # Done?
            if step.get("cmd_type") == "done":
                if step.get("done_summary"):
                    print()
                    print(step["done_summary"])
                print("DONE\n**********\n")
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
