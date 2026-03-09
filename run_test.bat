@echo off
cd /d "%~dp0"
pip install pytest
python -m pytest