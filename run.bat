@echo off
REM Ensure the script runs from its own directory for portability
cd /d "%~dp0"

REM Add src to PYTHONPATH for package imports
set PYTHONPATH=%~dp0src;%PYTHONPATH%

REM Launch AIOCR using Streamlit
poetry run streamlit run src/app/main.py %*

