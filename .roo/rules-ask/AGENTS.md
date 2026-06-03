# Ask Mode Rules

## Counterintuitive Structure
- **Root `main.py` is a dummy** — just prints "Hello from starting-codebase!"; real entrypoint is `backend/app.py`
- **`docs/` is at project root**, but the code loads it from `../docs` relative to `backend/` — same path, just resolved differently
- **Frontend is served by FastAPI** as static files from `../frontend` (relative to `backend/`), not as a separate dev server

## Hidden Context
- Course document format is strict (headers must match exactly) — see [`backend/document_processor.py:97-103`](../backend/document_processor.py:97) for expected format
- Lesson links are optional and placed on the line immediately after the lesson marker
- `sentence-transformers==5.0.0` is a major version — unusual for this library which typically uses lower versions
- No API authentication exists — `TrustedHostMiddleware` is configured with `allowed_hosts=["*"]` (development only)
