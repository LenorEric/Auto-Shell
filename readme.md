# AI-Auto shell

Lenor

## Usage

```shell
python ./auto_shell.py config.yml
```

Verified on Python 3.14

When the prompt is `goal`, enter in natural language what you want to do.
Prefix the goal with `IA ` to enter interactive mode (for example, `IA list recent logs`).
In interactive mode, the assistant will not end the session; use `[q]` when you decide the goal is achieved.

The console will first output the command to be executed, the command analysis, and the risk level, and will provide the reasons for doing so.

After that, the user can choose to execute the command `[y]`, edit the command`[e]`(unavailable for python code), reject the current command`[r]`, ask questions about the current command`[a]`, or run current command and provide additional instructions`[i]`, quit current conversation`[q]`.



Asking Question in English is recommended since the prompt is based on English.

This program is specialized for the ChatGPT API and tested based on ChatGPT 5.2; using other models may result in compatibility issues.

## Frequently Asked Questions

## 1. Garbled Characters

Q: Garbled characters in the console output, especially when the system uses other encodings in Windows.

A: Run `chcp 65001` before call the script

## 2. Using proxy

Q: I want to use proxy to connect to the API server

A: Export the proxy in the environment variables. 

For windows cmd shell, for example:

```bash
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890
python auto_shell.py config.yml
```

For linux bash:

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
python auto_shell.py config.yml
```
