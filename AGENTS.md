# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Run Commands

- **Start server** (must run from `backend/` directory):
  ```bash
  cd backend && uv run uvicorn app:app --reload --port 8000
  ```
- **Install dependencies**: `uv sync` (uses `uv`, not pip)
- **No test/lint framework exists** — no tests, no linter, no type checker configured
- Dependencies locked in `uv.lock` (managed by `uv`)

## Project-Specific Conventions

### Document Format
Course documents (`.txt`) must follow this exact header format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]
Lesson 1: [lesson title]
Lesson Link: [url]  (optional, next line after lesson marker)
```

### Architecture
- FastAPI serves frontend as static files from `../frontend` (relative to `backend/`)
- Docs loaded from `../docs` at startup (relative to `backend/`)
- ChromaDB persisted at `backend/chroma_db` (gitignored)
- Two ChromaDB collections: `course_catalog` (metadata) and `course_content` (chunks)
- Chunk prefix inconsistency: first chunk of each lesson uses `"Lesson X content:"`, subsequent chunks use `"Course Y Lesson X content:"` ([`backend/document_processor.py:186`](backend/document_processor.py:186) vs [`backend/document_processor.py:234`](backend/document_processor.py:234))
- Config via `@dataclass` singleton pattern ([`backend/config.py:27`](backend/config.py:27))
- Frontend renders markdown via `marked` library (loaded from CDN, not bundled)

### API
- `POST /api/query` — body: `{"query": "...", "session_id": null}`, returns `{answer, sources, session_id}`
- `GET /api/courses` — returns `{total_courses, course_titles}`
- AI uses Claude tool-calling with `search_course_content` tool (defined in [`backend/search_tools.py:27`](backend/search_tools.py:27))
