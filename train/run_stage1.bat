@echo off
REM Stage 1 pretrain (general English). Resumable: re-run to continue.
py "%~dp0trainer.py" --stage 1 >> "%~dp0..\stage1.log" 2>&1