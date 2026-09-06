@echo off
REM EDITH - a Marvel terminal.
REM
REM Not `where py ... && (py ...) || (python ...)`: that chains on the exit
REM code of the whole group, so pressing ctrl-c inside EDITH looked like "py
REM failed" and fell through to `python`, which on Windows is the Store stub -
REM "Python not found; run without arguments to install".
setlocal
where py >nul 2>nul
if errorlevel 1 goto :nopy
py "%~dp0infer\terminal.py" %*
exit /b %errorlevel%
:nopy
python "%~dp0infer\terminal.py" %*
exit /b %errorlevel%
