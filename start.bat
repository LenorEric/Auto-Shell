@echo off
chcp 65001 >nul
cd /d "%~dp0"
python auto_shell.py config.yml
pause
