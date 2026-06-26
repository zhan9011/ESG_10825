@echo off
setlocal
cd /d "%~dp0\.."
python -m src.inference.predict --config config/inference.yaml %*
