# AI-Auto shell

Lenor

## Usage

```shell
python ./auto_shell.py config.yml
```

Verified on Python 3.14

When the prompt is `goal`, enter in natural language what you want to do.

The console will first output the command to be executed, the command analysis, and the risk level, and will provide the reasons for doing so.

After that, the user can choose to execute the command `[y]`, edit the command`[e]`, skip the current command without executing it`[s]`, ask questions about the current command`[a]`, or skip the current command and provide additional requirements to generate a new command`[i]`.

## Frequently Asked Questions

Q: Garbled characters in the console output, especially when the system uses other encodings in Windows.

A: Run `chcp 65001` before call the script
