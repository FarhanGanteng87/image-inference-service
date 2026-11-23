# Production-Ready Image Inference Service

A modular REST API built with **FastAPI** that transitions computer vision preprocessing and inference workflows from isolated notebooks into a structured, testable backend service.

## Architecture Overview
- **`app/main.py`**: Exposes `/health` and `/predict` REST endpoints with automatic OpenAPI documentation.
- **`app/validator.py`**: Guards the service against production **"What If?" edge cases** (invalid MIME types, oversized files > 5 MB, corrupted binary payloads, and out-of-bounds image resolutions).
- **`app/predictor.py`**: Handles deterministic image resizing (`224x224`), tensor normalization, inference scoring, and latency measurement.
- **`tests/test_api.py`**: Automated test suite verifying both standard inference and failure scenarios.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` in your browser to test the `/predict` endpoint interactively via Swagger UI.

### 3. Run Automated Tests
```bash
pytest -v
```
