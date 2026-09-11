@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "%~dp0project-env.bat"
if errorlevel 1 goto failed
call uv sync --locked --link-mode copy
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -X utf8 -m qwenbrain.download
if errorlevel 1 goto failed
popd
echo Setup complete. Double-click chat.bat to start.
exit /b 0
:failed
popd
echo Setup failed. See the error above.
pause
exit /b 1
