# Code Mode Rules

## Critical Gotchas
- **Run from `backend/`**: All `uv` commands must execute from `backend/` directory (paths like `../docs` and `../frontend` are relative to `backend/`)
- **DeepSeek API**: Uses OpenAI-compatible SDK (`openai` package, not `anthropic`); config via `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` in `.env`
- **Chunk prefix inconsistency**: When modifying [`document_processor.py`](../backend/document_processor.py:186), note that first chunk uses `"Lesson X content:"` prefix but subsequent chunks use `"Course Y Lesson X content:"` — don't "fix" this without understanding both code paths
- **No linter/formatter**: Manually ensure PEP 8 compliance; there's no config enforcing it

## Coding Patterns
- Config uses `@dataclass` singleton at module level ([`backend/config.py:27`](../backend/config.py:27))
- Vector store uses two ChromaDB collections: `course_catalog` (metadata) and `course_content` (chunks)
- Search tool follows `Tool` ABC pattern in [`backend/search_tools.py:6`](../backend/search_tools.py:6); register new tools via `ToolManager.register_tool()`
- Lesson metadata stored as JSON string (`lessons_json`) in ChromaDB metadata field — must `json.loads()` when reading

## File Conventions
- Course documents (.txt) use strict header format: `Course Title:`, `Course Link:`, `Course Instructor:`, `Lesson X:` markers
- Frontend uses vanilla JS + `marked` (CDN) — no framework, no build step
