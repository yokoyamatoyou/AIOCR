# AIOCR

AIOCR is a small sample project demonstrating a Streamlit based OCR workflow. The application accepts images and ROI (Region of Interest) definitions, processes them with OpenCV, then displays OCR results in a simple web interface.  

## Setup

1. **Python**: Install Python 3.9 or later (3.10 recommended).
2. **Install dependencies** using [Poetry](https://python-poetry.org/):

   ```bash
   poetry install
   ```
   Alternatively you can install the packages directly with `pip` using the information in `pyproject.toml`.
3. **Environment variables**: copy `.env.example` to `.env` and update values such as `OPENAI_API_KEY` used by the application.

## Running the application

Start the Streamlit interface with:

```bash
streamlit run src/app/main.py
```
On Windows you can run `run.bat` which executes the same command.

This will launch a local web server where you can upload image files, a ZIP archive, or specify a local folder containing images
 and the ROI definition YAML. ZIP archives and folders are copied to a temporary directory and all contained images are processed sequentially.

## Running tests

Execute all unit tests with:

```bash
pytest
```

The tests cover key processing modules and ensure the Streamlit application functions as expected.
