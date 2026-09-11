@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "%~dp0project-env.bat"
if errorlevel 1 goto failed
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Run setup.bat first.
    goto failed
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m qwenbrain.check_update
set "updateExit=%errorlevel%"
popd
pause
exit /b %updateExit%
:failed
popd
pause
exit /b 1
