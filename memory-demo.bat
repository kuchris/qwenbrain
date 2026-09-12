@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project Python was not found. With Python 3.11 installed, run:
    echo python -m qwenbrain.memory_demo
    popd
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m qwenbrain.memory_demo %*
set "demoExit=%errorlevel%"
popd
exit /b %demoExit%
