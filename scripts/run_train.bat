@echo off
setlocal
cd /d "%~dp0\.."
python -m src.training.train --config config/train.yaml --stage all %*
