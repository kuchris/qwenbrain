@echo off
rem Call this before starting Python. Changes apply only to the calling process.
rem Do not use setlocal here: launchers need to inherit these settings.
set "HF_HOME=%~dp0.cache\huggingface"
set "HF_HUB_CACHE=%~dp0models\hub"
set "HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "HF_ASSETS_CACHE=%HF_HOME%\assets"
set "HF_DATASETS_CACHE=%~dp0data\cache"
set "TORCH_HOME=%~dp0.cache\torch"
set "TEMP=%~dp0.cache\tmp"
set "TMP=%TEMP%"
rem Clear the deprecated override so it cannot redirect model files elsewhere.
set "TRANSFORMERS_CACHE="
for %%D in ("%HF_HOME%" "%HF_HUB_CACHE%" "%HF_XET_CACHE%" "%HF_ASSETS_CACHE%" "%HF_DATASETS_CACHE%" "%TORCH_HOME%" "%TEMP%" "%~dp0adapters" "%~dp0checkpoints" "%~dp0data\conversations" "%~dp0runs" "%~dp0logs") do (
    if not exist "%%~D" mkdir "%%~D"
    if errorlevel 1 exit /b 1
)
exit /b 0
