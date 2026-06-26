@echo off
setlocal
cd /d "%~dp0\.."
python tests\test_preprocessing.py
python tests\test_training_entry.py
python tests\test_result_consistency.py
python tests\test_inference_without_probability_file.py
python tests\test_corrupt_probability_file.py
