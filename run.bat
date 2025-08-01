@echo off
REM Ensure the script runs from its own directory for portability
cd /d "%~dp0"

REM Launch AIOCR using Streamlit
poetry run streamlit run src/app/main.py %*

